/** Phase 1 types — Mini-Town */

export interface AgentState {
  id: string;
  name: string;
  age: number;
  occupation: string;
  personality: string;
  background: string;
  x: number;
  y: number;
  currentLocation: string;
  currentAction: string;
  actionId?: string;
  actionPhase?: string;
  actionStartedAt?: number;
  actionExpectedDuration?: number;
  actionProgress?: number;
  actionTarget?: string;
  actionReason?: string;
  mood: string;
  emoji: string;
  status: 'IDLE' | 'MOVING' | 'ACTING' | 'SPEAKING' | 'LISTENING';
  needs: { energy: number; hunger: number; social: number };
}

export interface Location {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  emoji: string;
}

export interface Memory {
  id: string;
  time: string;
  location: string;
  content: string;
  importance: number;
  type: string;
  tier?: 'working' | 'episodic' | 'core';
  emotion?: string;
  eventType?: string;
  unresolved?: boolean;
  futureIntention?: string;
  formationScore?: number;
}

export interface Belief {
  id: string;
  proposition: string;
  confidence: number;
  status: 'observed' | 'reported' | 'inferred' | 'uncertain' | 'disproven' | string;
  learned_at?: number;
  last_confirmed_at?: number;
  source_fact_ids: string[];
}

export interface Intention {
  id: string;
  description: string;
  source: string;
  status: string;
  confidence: number;
  public: boolean;
  source_ids: string[];
  target_location: string;
}

export interface MentalStateSummary {
  beliefs: Belief[];
  intentions: Intention[];
}

export interface RelationshipData {
  agentA: string;
  agentB: string;
  familiarity: number;
  affinity: number;
  trust: number;
  anchors: { event: string; valence: number; time: string }[];
  tags: string[];
  interactionCount: number;
  lastInteraction: string;
}

export interface Weather {
  condition: string;
  temperature: number;
  wind: number;
  description: string;
}

export type Infrastructure = Record<string, string>;

export interface TownEvent {
  type: string;
  location: string;
  description: string;
}

export interface Event {
  id: string;
  time: string;
  type: string;
  agentIds: string[];
  location: string;
  content: string;
  speaker?: string;
  simTime?: string;
  simTimestamp?: number;
  sequence?: number;
  interactionId?: string;
  parentEventId?: string;
  factId?: string;
  cause?: string;
  outcome?: string;
  dialogueId?: string;
  trigger?: { type: string; intentionId?: string; memoryId?: string };
}

export interface DialogueMessage {
  turn: number;
  simTime: string;
  speakerId: string;
  speakerName: string;
  content: string;
  action: 'continue' | 'end';
}

export interface InteractionRecord {
  id: string;
  sim_timestamp: number;
  state_version: number;
  initiator_id: string;
  target_type: string;
  target_id: string;
  interaction_type: string;
  intention_id: string;
  status: string;
  payload: Record<string, unknown>;
  expected_effects: Record<string, unknown>[];
  actual_effects: Record<string, unknown>[];
  fact_ids: string[];
}

export interface DialogueDetail {
  id: string;
  startedAt: string;
  endedAt: string;
  location: string;
  participants: string[];
  participantNames: string[];
  summary: string;
  valence: number;
  status: 'active' | 'completed';
  messages: DialogueMessage[];
}

export interface WorldEntity {
  id: string;
  name: string;
  kind: string;
  entity_type: string;
  location: string;
  quantity: number;
  availability: string;
  portable: boolean;
  capabilities: string[];
  components: Record<string, unknown>;
  state: Record<string, unknown>;
  lastChangedAt?: number;
  properties: Record<string, unknown>;
}

export interface WorldProcess {
  id: string;
  name: string;
  duration_minutes: number;
  required_capability?: string;
  inputs: { kind: string; quantity: number }[];
  outputs: { kind: string; quantity: number; name?: string }[];
}

export interface EnvironmentLocation {
  entities: WorldEntity[];
  processes: WorldProcess[];
}

export interface EnvironmentState {
  locations: Record<string, EnvironmentLocation>;
  inventories: Record<string, WorldEntity[]>;
}

export interface SceneZone {
  id: string;
  location: string;
  name: string;
  tags: string[];
  capacity: number;
}

export interface SceneAnchor {
  id: string;
  location: string;
  zone: string;
  name: string;
  x: number;
  y: number;
  capabilities: string[];
  capacity: number;
}

export interface SceneContainer {
  id: string;
  location: string;
  anchor: string;
  zone: string;
  name: string;
  capacity: number;
  usedCapacity: number;
  availableCapacity: number;
  acceptsKinds: string[];
}

export interface SceneRoute {
  id: string;
  name: string;
  type: string;
  endpointAnchorId: string;
  supportedResourceKinds: string[];
}

export interface SceneState {
  zones: SceneZone[];
  anchors: SceneAnchor[];
  containers: SceneContainer[];
  routes: SceneRoute[];
}

export interface SimulationTask {
  id: string;
  title: string;
  assignee_id: string;
  source: string;
  location_id: string;
  interaction_type: string;
  duration_minutes: number;
  earliest_at: number;
  deadline_at: number | null;
  priority: number;
  status: 'planned' | 'active' | 'blocked' | 'completed' | 'expired' | 'cancelled';
  blocking_reason: string;
}

export interface LogisticsOrder {
  id: string;
  resource_kind: string;
  quantity: number;
  destination_location: string;
  destination_container_id: string;
  route_id: string;
  requester_id: string;
  carrier_id: string;
  status: string;
  shipment_id: string;
  failure_reason: string;
}

export interface LogisticsShipment {
  id: string;
  order_id: string;
  resource_id: string;
  quantity: number;
  route_id: string;
  source_location: string;
  destination_location: string;
  destination_container_id: string;
  carrier_id: string;
  status: string;
  arrival_at: number | null;
  failure_reason: string;
}

export interface LogisticsState {
  orders: LogisticsOrder[];
  shipments: LogisticsShipment[];
  events: { type: string; order_id: string; shipment_id?: string; sim_timestamp: number }[];
}

export interface WorldState {
  world?: { id: string; name: string };
  map?: { width: number; height: number };
  time: { day: number; hour: number; minute: number };
  speed: number;
  running: boolean;
  stateVersion?: number;
  agents: AgentState[];
  locations: Location[];
  recentEvents: Event[];
  weather: Weather;
  infrastructure: Infrastructure;
  townEvents: TownEvent[];
  environment?: EnvironmentState;
  scene?: SceneState;
  tasks?: SimulationTask[];
  logistics?: LogisticsState;
  habits?: Record<string, Habit[]>;
}

export interface Habit {
  key: string;
  agentId: string;
  blockId: string;
  locationId: string;
  activity: string;
  strength: number;
  voluntaryCount: number;
}

export interface TraceEntry {
  id: number;
  time: string;
  simTime: string;
  agentId: string;
  layer: string;
  event: string;
  detail: string;
}
