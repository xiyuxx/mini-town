import { EnvironmentLocation, WorldEntity, WorldState } from '../types';

interface EnvironmentPanelProps {
  worldState: WorldState;
  agentId: string | null;
}

function EntityRow({ entity }: { entity: WorldEntity }) {
  const state = Object.entries(entity.state || {})
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(', ');
  return (
    <div className="environment-entity">
      <div className="environment-entity-main">
        <strong>{entity.name || entity.id}</strong>
        <span className="environment-kind">{entity.kind}</span>
        <span className={entity.availability === 'available' ? 'environment-ok' : 'environment-bad'}>
          {entity.availability}
        </span>
      </div>
      <div className="environment-meta">
        数量 {entity.quantity}
        {entity.capabilities.length > 0 && ` · 能力 ${entity.capabilities.join(', ')}`}
        {state && ` · ${state}`}
      </div>
    </div>
  );
}

export default function EnvironmentPanel({ worldState, agentId }: EnvironmentPanelProps) {
  if (!agentId || !worldState.environment) {
    return (
      <section className="environment-panel">
        <h4>🌐 世界环境</h4>
        <div className="environment-empty">选择角色后查看其可见环境</div>
      </section>
    );
  }

  const agent = worldState.agents.find(item => item.id === agentId);
  const locationId = agent?.currentLocation || '';
  const location = worldState.environment.locations[locationId] as EnvironmentLocation | undefined;
  const inventory = worldState.environment.inventories[agentId] || [];
  const anchors = (worldState.scene?.anchors || []).filter(anchor => anchor.location === locationId);
  const containers = (worldState.scene?.containers || []).filter(container => container.location === locationId);
  const routes = (worldState.scene?.routes || []).filter(route =>
    anchors.some(anchor => anchor.id === route.endpointAnchorId),
  );

  return (
    <section className="environment-panel">
      <div className="environment-heading">
        <h4>🌐 当前环境</h4>
        <span>{locationId || '未知位置'}</span>
      </div>
      <div className="environment-section-title">物流</div>
      {worldState.logistics?.orders?.length ? worldState.logistics.orders.map(order => {
        const shipment = worldState.logistics?.shipments.find(item => item.id === order.shipment_id);
        return (
          <div className="environment-process" key={order.id}>
            <strong>{order.resource_kind} x {order.quantity}</strong>
            <span>{order.status}</span>
            <small>
              {shipment ? `${shipment.status} · ${shipment.route_id}` : order.route_id}
              {order.failure_reason ? ` · ${order.failure_reason}` : ''}
            </small>
          </div>
        );
      }) : <div className="environment-empty">当前没有订单</div>}
      <div className="environment-section-title">交互点</div>
      {anchors.length ? anchors.map(anchor => (
        <div className="environment-process" key={anchor.id}>
          <strong>{anchor.name || anchor.id}</strong>
          <span>{anchor.capacity} 个位置</span>
          <small>{anchor.capabilities.join(' · ') || '通用交互'}</small>
        </div>
      )) : <div className="environment-empty">当前地点没有声明交互点</div>}
      <div className="environment-section-title">储存与路线</div>
      {containers.length ? containers.map(container => (
        <div className="environment-process" key={container.id}>
          <strong>{container.name || container.id}</strong>
          <span>{container.usedCapacity} / {container.capacity}</span>
          <small>{container.acceptsKinds.join(' · ') || '通用存放'}</small>
        </div>
      )) : null}
      {routes.map(route => (
        <div className="environment-process" key={route.id}>
          <strong>{route.name || route.id}</strong>
          <span>{route.type}</span>
          <small>{route.supportedResourceKinds.join(' · ') || '通用运输'}</small>
        </div>
      ))}
      {!containers.length && !routes.length && <div className="environment-empty">当前地点没有储存或路线设施</div>}
      <div className="environment-section-title">地点实体</div>
      {location?.entities.length ? location.entities.map(entity => (
        <EntityRow key={entity.id} entity={entity} />
      )) : <div className="environment-empty">没有可见实体</div>}
      <div className="environment-section-title">可用过程</div>
      {location?.processes.length ? location.processes.map(process => (
        <div className="environment-process" key={process.id}>
          <strong>{process.name || process.id}</strong>
          <span>{process.duration_minutes} 分钟</span>
          <small>
            {process.inputs.map(item => `${item.kind}×${item.quantity}`).join(' + ') || '无输入'}
            {' → '}
            {process.outputs.map(item => `${item.name || item.kind}×${item.quantity}`).join(' + ')}
          </small>
        </div>
      )) : <div className="environment-empty">当前没有可用过程</div>}
      <div className="environment-section-title">携带物</div>
      {inventory.length ? inventory.map(entity => (
        <EntityRow key={entity.id} entity={entity} />
      )) : <div className="environment-empty">没有携带物</div>}
    </section>
  );
}
