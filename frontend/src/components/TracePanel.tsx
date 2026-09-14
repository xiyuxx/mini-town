import { useState, useEffect, useRef } from 'react';
import { TraceEntry } from '../types';

const LAYER_COLORS: Record<string, string> = {
  state: '#9e9e9e',
  perceive: '#4fc3f7',
  tool: '#ffb74d',
  llm: '#64b5f6',
  action: '#e57373',
  dialogue: '#ba68c8',
};

export default function TracePanel({ running, stateVersion = 0 }: { running: boolean; stateVersion?: number }) {
  const [traces, setTraces] = useState<TraceEntry[]>([]);
  const listRef = useRef<HTMLDivElement>(null);
  const prevStateVersion = useRef(stateVersion);

  useEffect(() => {
    if (prevStateVersion.current > 0 && stateVersion === 0) {
      setTraces([]);
    }
    prevStateVersion.current = stateVersion;
  }, [stateVersion]);

  useEffect(() => {
    const fetchTraces = () => {
      fetch('/api/traces')
        .then(r => r.json())
        .then((data: TraceEntry[]) => setTraces(data))
        .catch(() => {});
    };
    fetchTraces();
    const interval = setInterval(fetchTraces, 1000);
    return () => clearInterval(interval);
  }, [running]);
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [traces]);

  return (
    <div className="trace-panel">
      <h4>🔍 推理追踪</h4>
      <div className="trace-list" ref={listRef}>
        {traces.length === 0 && (
          <div className="trace-empty">暂无追踪数据</div>
        )}
        {traces.map(t => (
          <div
            key={t.id}
            className="trace-entry"
            style={{ borderLeftColor: LAYER_COLORS[t.layer] || '#888' }}
          >
            <div className="trace-header">
              <span
                className="trace-layer"
                style={{ color: LAYER_COLORS[t.layer] || '#888' }}
              >
                {t.layer}
              </span>
              <span className="trace-sim-time">{t.simTime}</span>
            </div>
            <div className="trace-event">{t.event}</div>
            <div className="trace-detail">{t.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
