import { useEffect, useMemo, useRef, useState } from 'react';
import { Event } from '../types';

interface DialogueFeedProps {
  events: Event[];
  onOpenDialogue: (dialogueId: string) => void;
}

type FeedFilter = 'all' | 'movement' | 'activity' | 'dialogue';

const FILTER_TYPES: Record<Exclude<FeedFilter, 'all'>, string[]> = {
  movement: ['move_start', 'movement_arrived'],
  activity: ['activity_started', 'activity_completed'],
  dialogue: ['dialogue_start', 'dialogue_line', 'dialogue_end'],
};

const VISIBLE_TYPES = Object.values(FILTER_TYPES).flat();

const CAUSE_LABELS: Record<string, string> = {
  commitment: '日程',
  routine: '日常安排',
  routine_block: '弹性安排',
  life_plan: '生活计划',
  movement_completed: '移动完成',
  logistics: '物流',
};

export default function DialogueFeed({ events, onOpenDialogue }: DialogueFeedProps) {
  const [filter, setFilter] = useState<FeedFilter>('all');
  const [following, setFollowing] = useState(true);
  const listRef = useRef<HTMLDivElement | null>(null);

  const displayed = useMemo(() => {
    const visible = events
      .filter(event => VISIBLE_TYPES.includes(event.type))
      .slice()
      .sort((a, b) => {
        const time = (a.simTimestamp || 0) - (b.simTimestamp || 0);
        return time || (a.sequence || 0) - (b.sequence || 0);
      });
    if (filter === 'all') return visible;
    return visible.filter(event => FILTER_TYPES[filter].includes(event.type));
  }, [events, filter]);

  useEffect(() => {
    if (!following || !listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [displayed, following]);

  const handleScroll = () => {
    const list = listRef.current;
    if (!list) return;
    const distanceFromBottom = list.scrollHeight - list.scrollTop - list.clientHeight;
    setFollowing(distanceFromBottom < 24);
  };

  const jumpToLatest = () => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
    setFollowing(true);
  };

  return (
    <section className="dialogue-feed">
      <div className="feed-heading">
        <h4>小镇动态</h4>
        <div className="feed-filters" aria-label="动态类型">
          {(['all', 'movement', 'activity', 'dialogue'] as FeedFilter[]).map(value => (
            <button
              key={value}
              className={filter === value ? 'active' : ''}
              onClick={() => setFilter(value)}
            >
              {value === 'all' ? '全部' : value === 'movement' ? '移动' : value === 'activity' ? '活动' : '对话'}
            </button>
          ))}
        </div>
      </div>
      <div className="dialogue-feed-list" ref={listRef} onScroll={handleScroll}>
        {displayed.length === 0 && <div className="empty-state">暂无动态</div>}
        {displayed.map(event => {
          const clickable = Boolean(event.dialogueId && event.type.startsWith('dialogue_'));
          return (
            <button
              key={event.id}
              type="button"
              className={`event-item ${event.type} ${clickable ? 'clickable' : ''}`}
              onClick={() => clickable && onOpenDialogue(event.dialogueId!)}
              disabled={!clickable}
            >
              <span className="evt-time">{event.time}</span>
              {event.type === 'move_start' && <span className="evt-kind">出发</span>}
              {event.type === 'movement_arrived' && <span className="evt-kind">抵达</span>}
              {event.type === 'activity_started' && <span className="evt-kind">开始</span>}
              {event.type === 'activity_completed' && <span className="evt-kind">完成</span>}
              <span>{event.content}</span>
              {event.cause && <span className="evt-kind">{CAUSE_LABELS[event.cause] || event.cause}</span>}
              {event.outcome && <span className="evt-kind">{event.outcome}</span>}
              {event.trigger?.type === 'memory_intention' && <span className="evt-kind">记忆触发</span>}
            </button>
          );
        })}
      </div>
      {!following && (
        <button className="feed-latest" onClick={jumpToLatest}>回到最新</button>
      )}
    </section>
  );
}
