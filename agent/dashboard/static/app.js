const state = {
  snapshot: null,
};

const qs = (selector) => document.querySelector(selector);

function text(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function pct(value) {
  const number = Number(value || 0);
  return `${Math.round(number * 100)}%`;
}

function age(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

function cls(value) {
  return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "");
}

function metric(label, value, hint) {
  return `<div class="metric"><strong>${text(value)}</strong><span>${label}${hint ? ` · ${hint}` : ""}</span></div>`;
}

function tag(value) {
  return `<span class="tag ${cls(value)}">${text(value)}</span>`;
}

function empty(message) {
  return `<p class="empty">${message}</p>`;
}

async function loadSnapshot() {
  qs("#last-updated").textContent = "불러오는 중";
  const response = await fetch("/api/snapshot?limit=12", { cache: "no-store" });
  if (!response.ok) throw new Error(`snapshot failed: ${response.status}`);
  state.snapshot = await response.json();
  render(state.snapshot);
}

function render(data) {
  renderHero(data);
  renderMetrics(data);
  renderProcesses(data.processes || {});
  renderApprovals(data.approvals || {});
  renderActions(data);
  renderGoals(data.goals || {});
  renderSelfMap(data.self_map || {});
  renderIntelligence(data);
  renderEvents(data.events || {});
  qs("#last-updated").textContent = `업데이트 ${new Date().toLocaleTimeString("ko-KR")}`;
}

function renderHero(data) {
  const summary = data.summary || {};
  const metrics = data.metrics || {};
  const ring = qs("#health-ring");
  ring.className = `status-ring ${cls(summary.health)}`;
  ring.textContent = text(summary.health);
  qs("#headline").textContent = text(summary.headline, "Core 상태 없음");
  qs("#focus").textContent = text(summary.focus, "현재 focus 없음");
  qs("#profile").textContent = text(summary.profile);
  qs("#eval").textContent = `${text(metrics.last_eval_result)} / ${text(metrics.last_eval_score)}`;
  qs("#schema").textContent = text(data.schema_version);
}

function renderMetrics(data) {
  const m = data.metrics || {};
  qs("#metric-grid").innerHTML = [
    metric("대기 사용자 작업", m.queued_user_tasks_count),
    metric("대기 자율 작업", m.queued_autonomous_tasks_count),
    metric("승인 대기", m.pending_approvals_count),
    metric("Action 성공률", pct(m.action_execution_success_rate_24h), "24h 실행 기준"),
    metric("메모리", m.memories_count, `vector ${pct(m.memory_vector_coverage)}`),
    metric("Discord", m.discord_messages_24h, "24h 메시지"),
  ].join("");
}

function renderProcesses(processes) {
  const counts = processes.counts || {};
  qs("#process-counts").textContent = `running ${counts.running || 0} · waiting ${counts.waiting || 0} · terminal ${counts.terminal || 0}`;
  const items = processes.items || [];
  if (!items.length) {
    qs("#processes").innerHTML = `<tr><td colspan="6" class="empty">표시할 process가 없어.</td></tr>`;
    return;
  }
  qs("#processes").innerHTML = items.map((item) => {
    const progress = item.plan && item.plan.progress ? item.plan.progress : {};
    const ratio = Number(progress.ratio || 0);
    return `
      <tr>
        <td>${text(item.pid)}</td>
        <td>${tag(item.state)}<br><small>${age(item.age_seconds)}</small></td>
        <td>${text(item.queue)}</td>
        <td>${text(item.title)}</td>
        <td>
          <div class="progress" aria-label="진행률"><div style="width:${Math.max(0, Math.min(100, ratio * 100))}%"></div></div>
          <small>${text(progress.done, 0)} / ${text(progress.total, 0)}</small>
        </td>
        <td>${text(item.next)}</td>
      </tr>`;
  }).join("");
}

function renderApprovals(approvals) {
  qs("#approval-count").textContent = `${approvals.pending_count || 0}건`;
  const items = approvals.items || [];
  qs("#approvals").innerHTML = items.length ? items.map((item) => `
    <article class="item">
      <div class="item-title"><strong>#${item.id} ${text(item.action_type)}</strong>${tag(item.risk_level)}</div>
      <p>${text(item.description)}</p>
      <p>${text(item.reason, "승인 채널에서 처리 필요")} · ${age(item.age_seconds)}</p>
    </article>
  `).join("") : empty("승인 대기 없음.");
}

function renderActions(data) {
  const m = data.metrics || {};
  const actions = data.actions || {};
  qs("#action-rate").textContent = `실행 성공률 ${pct(m.action_execution_success_rate_24h)}`;
  const items = actions.items || [];
  qs("#actions").innerHTML = items.length ? items.map((item) => `
    <article class="item">
      <div class="item-title"><strong>#${item.id} ${text(item.purpose)}</strong>${tag(item.status)}</div>
      <p>${text(item.command_summary)}</p>
      <p>${text(item.profile)} · ${text(item.risk_level)} · rc=${text(item.returncode, "-")} · ${age(item.age_seconds)}</p>
    </article>
  `).join("") : empty("최근 action 없음.");
}

function renderGoals(goals) {
  const items = goals.items || [];
  qs("#goal-count").textContent = `${items.length}개`;
  qs("#goals").innerHTML = items.length ? items.slice(0, 6).map((item) => `
    <article class="item">
      <div class="item-title"><strong>#${item.id} ${text(item.title)}</strong>${tag(item.status)}</div>
      <p>${text(item.description, item.goal_type)}</p>
      <p>priority ${text(item.priority)} · ${text(item.risk_level)}</p>
    </article>
  `).join("") : empty("열린 목표 없음.");
}

function renderSelfMap(selfMap) {
  const available = selfMap && selfMap.available !== false;
  qs("#self-map-state").textContent = available ? "available" : "stale";
  const snapshot = selfMap.snapshot || {};
  qs("#self-map").innerHTML = `
    <div class="kv">
      <div><span>요약</span><span>${text(selfMap.summary, selfMap.reason)}</span></div>
      <div><span>호스트</span><span>${text(snapshot.hostname)}</span></div>
      <div><span>OS</span><span>${text(snapshot.os)}</span></div>
      <div><span>작업 위치</span><span>${text(snapshot.project_root)}</span></div>
      <div><span>갱신</span><span>${text(selfMap.created_at)}</span></div>
    </div>`;
}

function renderIntelligence(data) {
  const memory = data.memory || {};
  const skills = data.skills || {};
  const vector = memory.vector_status || {};
  qs("#vector-state").textContent = `coverage ${pct(vector.coverage || data.metrics?.memory_vector_coverage)}`;
  const memoryItems = (memory.hygiene_candidates || []).slice(0, 3).map((item) => `
    <article class="item">
      <div class="item-title"><strong>memory #${item.id || "-"}</strong>${tag(item.reason || "review")}</div>
      <p>${text(item.title || item.summary)}</p>
    </article>`);
  const skillItems = (skills.candidates || []).slice(0, 3).map((item) => `
    <article class="item">
      <div class="item-title"><strong>${text(item.name || item.title || "skill candidate")}</strong>${tag(item.status || "candidate")}</div>
      <p>${text(item.reason || item.trigger_description || item.summary)}</p>
    </article>`);
  qs("#intelligence").innerHTML = [...memoryItems, ...skillItems].join("") || empty("정리 후보 없음.");
}

function renderEvents(events) {
  const items = events.items || [];
  qs("#events").innerHTML = items.length ? items.map((item) => `
    <article class="item">
      <div class="item-title"><strong>#${item.id} ${text(item.event_type)}</strong><span class="tag">${text(item.source)}</span></div>
      <p>${text(item.content)}</p>
      <p>${text(item.ts)} · importance ${text(item.importance)}</p>
    </article>
  `).join("") : empty("최근 이벤트 없음.");
}

qs("#refresh").addEventListener("click", () => loadSnapshot().catch(showError));

function showError(error) {
  qs("#last-updated").textContent = "연결 실패";
  qs("#headline").textContent = "대시보드 스냅샷을 못 불러왔어";
  qs("#focus").textContent = error.message;
}

loadSnapshot().catch(showError);
setInterval(() => loadSnapshot().catch(showError), 15000);
