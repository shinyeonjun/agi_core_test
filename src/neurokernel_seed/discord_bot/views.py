from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .bot_utils import call_blocking as _call
from .bot_utils import interaction_user_id as _interaction_user_id
from .commands import format_code_block
from .core_client import CoreClient


ActivationVerifyNote = Callable[[Any], str]


@dataclass(frozen=True)
class DiscordViewFactories:
    proposal: Callable[[str], Any]
    work: Callable[[str], Any]
    activation: Callable[[str], Any]
    retry: Callable[[str], Any]
    promote: Callable[[str], Any]


def build_view_factories(
    *,
    discord: Any,
    core: CoreClient,
    allowed_user_ids: tuple[int, ...],
    activation_verify_note: ActivationVerifyNote,
) -> DiscordViewFactories:
    class AllowedInteractionMixin:
        async def _allowed(self, interaction: Any) -> bool:
            user_id = int(getattr(getattr(interaction, "user", None), "id", 0) or 0)
            if allowed_user_ids and user_id not in allowed_user_ids:
                await interaction.response.send_message("이 버튼은 허용된 사용자만 누를 수 있어.", ephemeral=True)
                return False
            return True

        async def _ack(self, interaction: Any) -> None:
            if not interaction.response.is_done():
                await interaction.response.defer()

        async def _edit_original(self, interaction: Any, content: str, *, view: Any | None = None) -> None:
            await interaction.edit_original_response(content=content, view=view)

        async def _send_error(self, interaction: Any, label: str, exc: Exception) -> None:
            content = f"{label}: `{type(exc).__name__}: {exc}`"
            if interaction.response.is_done():
                await interaction.followup.send(content, ephemeral=True)
                return
            await interaction.response.send_message(content, ephemeral=True)

    class ProposalReviewView(AllowedInteractionMixin, discord.ui.View):
        def __init__(self, proposal_id: str):
            super().__init__(timeout=60 * 60 * 24)
            self.proposal_id = proposal_id

        async def _transition(self, interaction: Any, status: str, path: str, label: str) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                payload = await _call(core.post, f"/capability-proposals/{self.proposal_id}/{path}", {"actor": _interaction_user_id(interaction)})
                proposal = payload.get("proposal", {}) if isinstance(payload, dict) else {}
                name = proposal.get("capability_name") or "능력 후보"
                await self._edit_original(interaction, f"{label}: {name}\n상태: `{status}`", view=None)
            except Exception as exc:
                await self._send_error(interaction, "처리 실패", exc)

        @discord.ui.button(label="개발 후보 승인", style=discord.ButtonStyle.success)
        async def approve_dev(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "approved_for_dev", "approve-dev", "좋아, 개발 후보로 올려뒀어")

        @discord.ui.button(label="보류", style=discord.ButtonStyle.secondary)
        async def defer(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "deferred", "defer", "일단 보류해둘게")

        @discord.ui.button(label="거절", style=discord.ButtonStyle.danger)
        async def reject(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "rejected", "reject", "후보를 거절 처리했어")

    class WorkReviewView(AllowedInteractionMixin, discord.ui.View):
        def __init__(self, work_id: str):
            super().__init__(timeout=60 * 60 * 24)
            self.work_id = work_id

        async def _transition(self, interaction: Any, status: str, label: str) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                payload = await _call(core.post, f"/work-items/{self.work_id}/status", {"status": status, "actor": _interaction_user_id(interaction)})
                item = payload.get("work_item", {}) if isinstance(payload, dict) else {}
                title = item.get("title") or "work"
                await self._edit_original(interaction, f"{label}: {title}\n상태: `{status}`", view=None)
            except Exception as exc:
                await self._send_error(interaction, "처리 실패", exc)

        @discord.ui.button(label="작업 승인", style=discord.ButtonStyle.success)
        async def accept_work(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "accepted", "작업을 승인했어")

        @discord.ui.button(label="보류", style=discord.ButtonStyle.secondary)
        async def defer_work(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "deferred", "작업을 보류했어")

        @discord.ui.button(label="거절", style=discord.ButtonStyle.danger)
        async def reject_work(self, interaction: Any, button: Any) -> None:
            await self._transition(interaction, "rejected", "작업을 거절했어")

    class ActivationReviewView(AllowedInteractionMixin, discord.ui.View):
        def __init__(self, work_id: str):
            super().__init__(timeout=60 * 60 * 24)
            self.work_id = work_id

        @discord.ui.button(label="패치 장착 승인", style=discord.ButtonStyle.success)
        async def activate(self, interaction: Any, button: Any) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                payload = await _call(core.post, f"/work-items/{self.work_id}/activate", {"actor": _interaction_user_id(interaction)})
                action = (payload.get("action_id") or payload.get("proposal_id") or self.work_id) if isinstance(payload, dict) else self.work_id
                reload_note = "\n서비스 재시작이 필요해." if isinstance(payload, dict) and payload.get("service_reload_required") else ""
                verify_note = activation_verify_note(payload)
                await self._edit_original(interaction, f"장착 완료: `{action}`{verify_note}{reload_note}", view=None)
            except Exception as exc:
                await self._send_error(interaction, "장착 실패", exc)

        @discord.ui.button(label="수정 필요", style=discord.ButtonStyle.secondary)
        async def needs_review(self, interaction: Any, button: Any) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                await _call(core.post, f"/work-items/{self.work_id}/status", {"status": "reviewing", "actor": _interaction_user_id(interaction), "reason": "activation review requested"})
                await self._edit_original(interaction, "장착 보류. 수정/리뷰 상태로 돌려둘게.", view=None)
            except Exception as exc:
                await self._send_error(interaction, "처리 실패", exc)

    class WorkRetryView(AllowedInteractionMixin, discord.ui.View):
        def __init__(self, work_id: str):
            super().__init__(timeout=60 * 60 * 24)
            self.work_id = work_id

        @discord.ui.button(label="수정 재시도", style=discord.ButtonStyle.primary)
        async def retry(self, interaction: Any, button: Any) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                payload = await _call(core.post, f"/work-items/{self.work_id}/retry", {"actor": _interaction_user_id(interaction)})
                job = payload.get("job", {}) if isinstance(payload, dict) else {}
                if isinstance(payload, dict) and payload.get("queued"):
                    await self._edit_original(interaction, f"수정 재시도를 시작했어.\njob: `{job.get('job_id')}`", view=None)
                    return
                reason = payload.get("reason") if isinstance(payload, dict) else "unknown"
                await self._edit_original(interaction, f"재시도 시작 실패: `{reason}`", view=None)
            except Exception as exc:
                await self._send_error(interaction, "재시도 실패", exc)

    class WorkPromoteView(AllowedInteractionMixin, discord.ui.View):
        def __init__(self, work_id: str):
            super().__init__(timeout=60 * 60 * 24)
            self.work_id = work_id

        @discord.ui.button(label="개발 작업으로 전환", style=discord.ButtonStyle.primary)
        async def promote(self, interaction: Any, button: Any) -> None:
            if not await self._allowed(interaction):
                return
            await self._ack(interaction)
            try:
                payload = await _call(core.post, f"/work-items/{self.work_id}/promote-self-patch", {"actor": _interaction_user_id(interaction)})
                child = payload.get("child_work_item", {}) if isinstance(payload, dict) else {}
                queue = payload.get("queue", {}) if isinstance(payload, dict) else {}
                if isinstance(payload, dict) and child:
                    queued = "큐에 들어갔어" if isinstance(queue, dict) and queue.get("queued") else f"큐 대기 실패: {queue.get('reason') if isinstance(queue, dict) else 'unknown'}"
                    await self._edit_original(interaction, f"개발 작업으로 전환했어: {child.get('title') or self.work_id}\nchild: `{child.get('work_id')}`\n{queued}", view=None)
                    return
                await self._edit_original(interaction, format_code_block(payload), view=None)
            except Exception as exc:
                await self._send_error(interaction, "전환 실패", exc)

    return DiscordViewFactories(
        proposal=ProposalReviewView,
        work=WorkReviewView,
        activation=ActivationReviewView,
        retry=WorkRetryView,
        promote=WorkPromoteView,
    )
