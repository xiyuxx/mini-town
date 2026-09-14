import { useState, useEffect, useCallback } from 'react';
import { WorldState, AgentState, Memory, MentalStateSummary } from './types';
import TownMap from './components/TownMap';
import AgentPanel from './components/AgentPanel';
import DialogueFeed from './components/DialogueFeed';
import DialogueDrawer from './components/DialogueDrawer';
import TracePanel from './components/TracePanel';
import EnvironmentPanel from './components/EnvironmentPanel';
import TimeControls from './components/TimeControls';
import { useSimulation } from './hooks/useSimulation';
import { useToast } from './hooks/useToast';

export default function App() {
  const worldState = useSimulation();
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [agentMemories, setAgentMemories] = useState<Memory[]>([]);
  const [mentalState, setMentalState] = useState<MentalStateSummary | null>(null);
  const [selectedDialogueId, setSelectedDialogueId] = useState<string | null>(null);
  const { toasts, toast } = useToast();
  const selectedAgent = worldState?.agents.find(a => a.id === selectedAgentId) ?? null;
  const locationNames = Object.fromEntries((worldState?.locations || []).map(location => [location.id, location.name]));

  const selectAgent = useCallback((id: string | null) => {
    setSelectedAgentId(id);
    if (!id) {
      setAgentMemories([]);
      setMentalState(null);
      return;
    }
    fetch(`/api/agents/${id}`)
      .then(r => r.json())
      .then(d => {
        setAgentMemories(d.memories || []);
        setMentalState(d.mentalState || null);
      })
      .catch(() => {
        setAgentMemories([]);
        setMentalState(null);
      });
  }, []);

  // Refresh memories when selected agent changes
  useEffect(() => {
    if (selectedAgentId) {
      const interval = setInterval(() => {
        fetch(`/api/agents/${selectedAgentId}`)
          .then(r => r.json())
          .then(d => {
            setAgentMemories(d.memories || []);
            setMentalState(d.mentalState || null);
          })
          .catch(() => {});
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [selectedAgentId]);

  if (!worldState) {
    return (
      <div className="loading">
        <h1>🏘️ Mini Town</h1>
        <p>正在连接服务器...</p>
      </div>
    );
  }

  return (
    <div className="app">
      <TimeControls
        time={worldState.time}
        speed={worldState.speed}
        running={worldState.running}
        onToast={toast}
      />
      <div className="main-content">
        <div className="map-panel">
          <TownMap
            locations={worldState.locations}
            agents={worldState.agents}
            selectedAgentId={selectedAgentId}
            recentEvents={worldState.recentEvents}
            worldState={worldState}
            onSelectAgent={selectAgent}
          />
        </div>
        <div className="side-panel">
          <EnvironmentPanel worldState={worldState} agentId={selectedAgentId} />
          <AgentPanel
            agent={selectedAgent}
            memories={agentMemories}
            running={worldState.running}
            locationNames={locationNames}
            habits={selectedAgentId ? (worldState.habits?.[selectedAgentId] || []) : []}
            mentalState={mentalState}
            stateVersion={worldState.stateVersion}
          />
          <DialogueFeed
            events={worldState.recentEvents}
            onOpenDialogue={setSelectedDialogueId}
          />
          <TracePanel running={worldState.running} stateVersion={worldState.stateVersion} />
        </div>
      </div>
      <DialogueDrawer
        dialogueId={selectedDialogueId}
        onClose={() => setSelectedDialogueId(null)}
      />
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className="toast">{t.message}</div>
        ))}
      </div>
    </div>
  );
}
