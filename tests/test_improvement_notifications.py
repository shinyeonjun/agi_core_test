from neurokernel_seed.discord_bot.work_status import format_work_notification


def test_proposed_self_patch_notification_uses_capability_proposal_view():
    text, view_kind = format_work_notification(
        {
            "work_item": {
                "work_id": "work_active_model",
                "type": "self_patch",
                "title": "Read active model status",
                "status": "proposed",
            },
            "capability_proposal": {"proposal_id": "prop_active_model"},
            "events": [],
        }
    )

    assert "Read active model status" in text
    assert view_kind == "proposal:prop_active_model"
