import { useState, useEffect, useRef } from 'react';
import { AgentState, Habit, Memory, MentalStateSummary, RelationshipData } from '../types';
interface AgentPanelProps {
  agent: AgentState | null;
  memories: Memory[];
  running: boolean;
  locationNames?: Record<string, string>;
  habits?: Habit[];
  mentalState?: MentalStateSummary | null;
  stateVersion?: number;
}

function phaseLabel(agent: AgentState): string {
  if (agent.actionPhase) return agent.actionPhase;
  if (agent.status === 'MOVING') return '前往中';
  if (agent.status === 'ACTING') return '执行中';
  if (agent.status === 'SPEAKING' || agent.status === 'LISTENING') return '交流中';
  return '待决定';
}

function formatProgress(agent: AgentState): string {
  if (typeof agent.actionProgress !== 'number') return '';
  return `${Math.round(agent.actionProgress * 100)}%`;
}

function formatNumber(value: number, digits = 1, signed = false): string {
  const rounded = Number(value.toFixed(digits));
  const normalized = Object.is(rounded, -0) ? 0 : rounded;
  return `${signed && normalized > 0 ? '+' : ''}${normalized.toFixed(digits)}`;
}

function extractTime(time: string): string {
  if (!time) return '';
  // "第1天 06:15" → extract "06:15"
  const parts = time.split(' ');
  return parts.length > 1 ? parts[1] : time.slice(0, 5);
}

export default function AgentPanel({ agent, memories, running, locationNames = {}, habits = [], mentalState = null, stateVersion = 0 }: AgentPanelProps) {
  const [relationships, setRelationships] = useState<RelationshipData[]>([]);

  const prevStateVersion = useRef(stateVersion);

  useEffect(() => {
    if (prevStateVersion.current > 0 && stateVersion === 0) {
      setRelationships([]);
    }
    prevStateVersion.current = stateVersion;

    if (!agent) return;
    fetch(`/api/agents/${agent.id}/relationships`)
      .then(r => r.json())
      .then((data: RelationshipData[]) => setRelationships(data))
      .catch(() => setRelationships([]));
  }, [agent, running]);

  if (!agent) {
    return (
      <div className="agent-panel">
        <div className="placeholder">👆 点击地图上的角色查看详情</div>
      </div>
    );
  }

  return (
    <div className="agent-panel">
      <div className="header">
        <span className="avatar">{agent.emoji}</span>
        <div className="info">
          <h3>{agent.name}</h3>
          <div className="meta">{agent.age}岁 · {agent.occupation}</div>
        </div>
      </div>

      <div className="status">
        <span>📍 {locationNames[agent.currentLocation] || agent.currentLocation}</span>
        <span>{agent.mood === '愉快' ? '😊' : agent.mood === '低落' ? '😞' : '😐'} {agent.mood}</span>
      </div>

      <div className="agent-action-card">
        <div className="agent-action-heading">
          <span className={`action-status action-${agent.status.toLowerCase()}`}>{phaseLabel(agent)}</span>
          {formatProgress(agent) && <span className="action-progress-text">{formatProgress(agent)}</span>}
        </div>
        <div className="agent-action-text">{agent.currentAction || '等待下一步决定'}</div>
        {agent.actionTarget && <div className="agent-action-meta">目标：{agent.actionTarget}</div>}
        {agent.actionReason && <div className="agent-action-meta">原因：{agent.actionReason}</div>}
        {typeof agent.actionProgress === 'number' && (
          <div className="agent-progress-track"><div style={{ width: `${Math.max(0, Math.min(100, agent.actionProgress * 100))}%` }} /></div>
        )}
      </div>

      <div className="bg">
        <strong>性格：</strong>{agent.personality}<br />
        <strong>背景：</strong>{agent.background}
      </div>

      <div className="memories">
        <h4>📝 最近记忆</h4>
        {memories.length === 0 && <div style={{ color: '#666', fontSize: '0.78rem' }}>暂无记忆</div>}
        {memories.slice(0, 20).map(m => (
          <div key={m.id} className={`mem-item tier-${m.tier || 'episodic'}`}>
            <div className="mem-meta">
              <span className="mem-time">{extractTime(m.time)}</span>
              <span className="mem-tier">{m.tier === 'core' ? '核心' : m.tier === 'working' ? '短期' : '情景'}</span>
              {m.emotion && <span className="mem-emotion">{m.emotion}</span>}
            </div>
            <div>{m.content}</div>
            {m.unresolved && m.futureIntention && (
              <div className="mem-intention">待处理：{m.futureIntention}</div>
            )}
          </div>
        ))}
      </div>

      <div className="memories">
        <h4>🧠 当前信念</h4>
        {!mentalState?.beliefs?.length && <div style={{ color: '#666', fontSize: '0.78rem' }}>暂无结构化信念</div>}
        {mentalState?.beliefs?.slice(-5).reverse().map(belief => (
          <div key={belief.id} className="mem-item">
            <div>{belief.proposition}</div>
            <div className="mem-meta">
              {belief.status === 'uncertain' ? '不确定' : belief.status} · 置信度 {Math.round(belief.confidence * 100)}%
              {belief.source_fact_ids.length > 0 && ` · 来源事实 ${belief.source_fact_ids.slice(0, 2).join(', ')}${belief.source_fact_ids.length > 2 ? '…' : ''}`}
            </div>
          </div>
        ))}
      </div>

      <div className="memories">
        <h4>🎯 未完成意图</h4>
        {!mentalState?.intentions?.length && <div style={{ color: '#666', fontSize: '0.78rem' }}>暂无未完成意图</div>}
        {mentalState?.intentions?.slice(-5).reverse().map(intention => (
          <div key={intention.id} className="mem-item">
            <div>{intention.description}</div>
            <div className="mem-meta">{intention.source} · {intention.status} · 来源 {intention.source_ids.slice(0, 2).join(', ') || '无'}</div>
          </div>
        ))}
      </div>

      <div className="memories">
        <h4>🔁 行为倾向</h4>
        {habits.length === 0 && <div style={{ color: '#666', fontSize: '0.78rem' }}>尚未形成稳定倾向</div>}
        {habits.slice(0, 3).map(habit => (
          <div key={habit.key} className="mem-item">
            <div>{habit.activity}</div>
            <div className="mem-meta">{habit.blockId} · 倾向 {Math.round(habit.strength * 100)}%</div>
          </div>
        ))}
      </div>

      <div className="relationships">
        <h4>🤝 关系网络</h4>
        {relationships.length === 0 && (
          <div style={{ color: '#666', fontSize: '0.78rem' }}>暂无关系数据</div>
        )}
        {relationships.map(r => (
          <div key={`${r.agentA}-${r.agentB}`} className="rel-item">
            <div className="rel-header">
              <span className="rel-partner">{r.agentB}</span>
              <span className="rel-affinity" style={{ color: r.affinity >= 0 ? '#81c784' : '#e57373' }}>
                {formatNumber(r.affinity, 1, true)}
              </span>
            </div>
            <div className="rel-familiarity-bar">
              <div
                className="rel-familiarity-fill"
                style={{ width: `${Math.min(100, r.familiarity * 10)}%` }}
              />
            </div>
            <div className="rel-meta">
              熟悉度 {formatNumber(r.familiarity, 2)} · 信任 {formatNumber(r.trust, 2)} · 互动{r.interactionCount}次
            </div>
            {r.anchors.length > 0 && (
              <div className="rel-anchors">
                {r.anchors.slice(-3).map((a, i) => (
                  <div key={i} className="rel-anchor">
                    <span className="rel-anchor-valence" style={{ color: a.valence >= 0 ? '#81c784' : '#e57373' }}>
                      {formatNumber(a.valence, 1, true)}
                    </span>
                    <span className="rel-anchor-event">{a.event}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
