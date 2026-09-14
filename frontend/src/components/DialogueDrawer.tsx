import { useEffect, useState } from 'react';
import { DialogueDetail } from '../types';

interface DialogueDrawerProps {
  dialogueId: string | null;
  onClose: () => void;
}

export default function DialogueDrawer({ dialogueId, onClose }: DialogueDrawerProps) {
  const [dialogue, setDialogue] = useState<DialogueDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!dialogueId) {
      setDialogue(null);
      return;
    }
    setLoading(true);
    fetch(`/api/dialogues/${dialogueId}`)
      .then(response => {
        if (!response.ok) throw new Error('dialogue not found');
        return response.json();
      })
      .then(setDialogue)
      .catch(() => setDialogue(null))
      .finally(() => setLoading(false));
  }, [dialogueId]);

  if (!dialogueId) return null;

  return (
    <div className="dialogue-drawer-backdrop" onClick={onClose}>
      <aside className="dialogue-drawer" onClick={event => event.stopPropagation()}>
        <header className="dialogue-drawer-header">
          <div>
            <h3>{dialogue?.participantNames.join('、') || '对话记录'}</h3>
            {dialogue && (
              <div className="dialogue-drawer-meta">
                {dialogue.startedAt} · {dialogue.location}
              </div>
            )}
          </div>
          <button className="icon-button drawer-close" onClick={onClose} title="关闭">×</button>
        </header>
        <div className="dialogue-drawer-body">
          {loading && <div className="empty-state">正在读取对话...</div>}
          {!loading && !dialogue && <div className="empty-state">无法读取这场对话</div>}
          {dialogue?.messages.map(message => (
            <div key={`${message.turn}-${message.speakerId}`} className="dialogue-message">
              <div className="dialogue-message-meta">
                <strong>{message.speakerName}</strong>
                <span>{message.simTime}</span>
              </div>
              <div>{message.content}</div>
            </div>
          ))}
          {dialogue?.summary && (
            <div className="dialogue-summary">
              <strong>对话结果</strong>
              <div>{dialogue.summary}</div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
