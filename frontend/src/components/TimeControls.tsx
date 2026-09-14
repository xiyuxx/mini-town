import { useState, useEffect } from 'react';

interface TimeControlsProps {
  time: { day: number; hour: number; minute: number };
  speed: number;
  running: boolean;
  onToast: (msg: string) => void;
}

export default function TimeControls({ time, speed, running, onToast }: TimeControlsProps) {
  const [pending, setPending] = useState<string | null>(null);
  const [optimisticSpeed, setOptimisticSpeed] = useState(speed);

  useEffect(() => {
    setOptimisticSpeed(speed);
  }, [speed]);

  const request = async (path: string, body?: BodyInit) => {
    const response = await fetch(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body,
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  };

  const handleStart = async () => {
    onToast('\u25B6 模拟已开始');
    setPending('start');
    try { await request('/api/simulation/start'); } catch { onToast('启动失败'); }
    finally { setPending(null); }
  };

  const handlePause = async () => {
    onToast('\u23F8 模拟已暂停');
    setPending('pause');
    try { await request('/api/simulation/pause'); } catch { onToast('暂停失败'); }
    finally { setPending(null); }
  };

  const handleReset = async () => {
    onToast('\u21BA 已重置');
    setPending('reset');
    try {
      await request('/api/simulation/reset');
      setOptimisticSpeed(1);
    } catch { onToast('重置失败'); }
    finally { setPending(null); }
  };

  const handleSpeed = (s: number) => {
    onToast(`\u901F\u5EA6: ${s}x`);
    setOptimisticSpeed(s);
    request('/api/simulation/speed', JSON.stringify({ speed: s }))
      .catch(() => onToast('速度调整失败'));
  };

  const timeStr = `\u7B2C${time.day}\u5929 ${String(time.hour).padStart(2, '0')}:${String(time.minute).padStart(2, '0')}`;

  return (
    <div className="time-controls">
      <span className="title">{'\uD83C\uDFD8\uFE0F'} Mini Town</span>
      <span className="clock">{timeStr}</span>

      {!running ? (
        <button onClick={handleStart} disabled={pending !== null}>
          {pending === 'start' ? '\u23F3' : '\u25B6'} 开始
        </button>
      ) : (
        <button onClick={handlePause} disabled={pending !== null}>
          {pending === 'pause' ? '\u23F3' : '\u23F8'} 暂停
        </button>
      )}

      <button onClick={handleReset} disabled={pending !== null}>
        {pending === 'reset' ? '\u23F3' : '\u21BA'} 重置
      </button>

      <span className="speed-badge">速度:</span>
      {[0.5, 1, 2, 4].map(s => (
        <button
          key={s}
          onClick={() => handleSpeed(s)}
          className={optimisticSpeed === s ? 'active' : ''}
        >
          {s}x
        </button>
      ))}
    </div>
  );
}
