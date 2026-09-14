"""Deterministic, scenario-neutral order and shipment state machine."""

from dataclasses import asdict, dataclass, field

from .interactions import Resource, ResourceStore
from .scene import SceneIndex
from .tasks import Task, TaskStore


ORDER_STATES = {"requested", "confirmed", "shipping", "delivered", "failed", "cancelled"}
SHIPMENT_STATES = {"planned", "dispatched", "in_transit", "arrived", "received", "blocked", "failed"}


@dataclass
class Order:
    id: str
    resource_kind: str
    quantity: float
    destination_location: str
    destination_container_id: str
    route_id: str
    requester_id: str = ""
    supplier_id: str = ""
    carrier_id: str = ""
    priority: float = 0.5
    requested_at: int = 0
    deliver_by: int | None = None
    source_location: str = ""
    status: str = "requested"
    shipment_id: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shipment:
    id: str
    order_id: str
    resource_id: str
    quantity: float
    route_id: str
    source_location: str
    destination_location: str
    destination_container_id: str
    carrier_id: str = ""
    status: str = "planned"
    dispatched_at: int | None = None
    arrival_at: int | None = None
    received_at: int | None = None
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LogisticsEngine:
    """Advance supply contracts without network or LLM dependencies.

    External orders materialize a resource at the route endpoint on dispatch.
    Local resources are reserved by moving them to a shipment holding location;
    receipt then uses the existing container capacity and kind constraints.
    """

    def __init__(self, world_pack, resources: ResourceStore, scene: SceneIndex,
                 tasks: TaskStore):
        self.world_pack = world_pack
        self.resources = resources
        self.scene = scene
        self.tasks = tasks
        self.orders: dict[str, Order] = {}
        self.shipments: dict[str, Shipment] = {}
        self.events: list[dict] = []

    def reset(self) -> None:
        self.orders.clear()
        self.shipments.clear()
        self.events.clear()
        for raw in self.world_pack.data.get("orders", []):
            self.orders[str(raw["id"])] = Order(
                id=str(raw["id"]), resource_kind=str(raw["resource_kind"]),
                quantity=float(raw["quantity"]),
                destination_location=str(raw["destination_location"]),
                destination_container_id=str(raw["destination_container_id"]),
                route_id=str(raw["route_id"]), requester_id=str(raw.get("requester_id", "")),
                supplier_id=str(raw.get("supplier_id", "")),
                carrier_id=str(raw.get("carrier_id", raw.get("supplier_id", ""))),
                priority=float(raw.get("priority", 0.5)),
                requested_at=int(raw.get("requested_at", 0)),
                deliver_by=raw.get("deliver_by"), source_location=str(raw.get("source_location", "")),
            )

    def advance(self, now: int) -> list[dict]:
        self.events = []
        for order in self.orders.values():
            if order.status == "requested" and now >= order.requested_at:
                self._confirm(order, now)
            if order.status == "confirmed":
                self._plan_shipment(order, now)
            shipment = self.shipments.get(order.shipment_id)
            if shipment:
                self._advance_shipment(order, shipment, now)
        return list(self.events)

    def _confirm(self, order: Order, now: int) -> None:
        route = self.scene.routes.get(order.route_id)
        container = self.scene.containers.get(order.destination_container_id)
        if not route or order.resource_kind not in route.supported_resource_kinds:
            self._fail_order(order, "路线不支持该资源类型")
        elif not container or container.location_id != order.destination_location:
            self._fail_order(order, "交付容器不存在或位置不一致")
        elif container.accepts_kinds and order.resource_kind not in container.accepts_kinds:
            self._fail_order(order, "交付容器不接收该资源类型")
        elif order.quantity <= 0:
            self._fail_order(order, "订单数量必须为正数")
        else:
            order.status = "confirmed"
            self._event("order_confirmed", order, now)

    def _plan_shipment(self, order: Order, now: int) -> None:
        shipment_id = f"shipment:{order.id}"
        source = order.source_location or f"route:{order.route_id}"
        resource_id = f"shipment_resource:{order.id}"
        shipment = Shipment(
            id=shipment_id, order_id=order.id, resource_id=resource_id,
            quantity=order.quantity, route_id=order.route_id, source_location=source,
            destination_location=order.destination_location,
            destination_container_id=order.destination_container_id,
            carrier_id=order.carrier_id,
        )
        if order.source_location:
            source_items = self.resources.at_location(order.source_location, order.resource_kind)
            available = sum(item.quantity for item in source_items)
            if available < order.quantity:
                self._fail_order(order, "供给资源数量不足")
                return
            remaining = order.quantity
            for item in source_items:
                moved = min(item.quantity, remaining)
                item.quantity -= moved
                remaining -= moved
                if remaining <= 0:
                    break
            self.resources.add(Resource(
                id=resource_id, kind=order.resource_kind, location=source,
                quantity=order.quantity, name=order.resource_kind,
                portable=True, availability="available",
            ))
        else:
            self.resources.add(Resource(
                id=resource_id, kind=order.resource_kind, location=source,
                quantity=order.quantity, name=order.resource_kind,
                portable=True, availability="available",
            ))
        self.shipments[shipment_id] = shipment
        order.shipment_id = shipment_id
        order.status = "shipping"
        self._upsert_shipment_tasks(shipment, now)
        self._event("shipment_planned", order, now, shipment)

    def _advance_shipment(self, order: Order, shipment: Shipment, now: int) -> None:
        route = self.scene.routes.get(shipment.route_id)
        travel = self._travel_minutes(route)
        if shipment.status == "planned":
            shipment.status = "dispatched"
            shipment.dispatched_at = now
            shipment.arrival_at = now + travel
            shipment.status = "in_transit"
            self._upsert_shipment_tasks(shipment, now)
            self._event("shipment_dispatched", order, now, shipment)
        if shipment.status == "in_transit" and shipment.arrival_at is not None and now >= shipment.arrival_at:
            shipment.status = "arrived"
            resource = self.resources.get(shipment.resource_id)
            if resource:
                resource.location = shipment.destination_location
            self._upsert_shipment_tasks(shipment, now)
            self._event("shipment_arrived", order, now, shipment)
        if shipment.status == "arrived":
            container = self.scene.containers.get(shipment.destination_container_id)
            resource = self.resources.get(shipment.resource_id)
            usage = self.resources.container_usage(shipment.destination_container_id)
            if not container or not resource:
                self._fail_shipment(order, shipment, "收货资源或容器不存在")
            elif usage + shipment.quantity > container.capacity:
                shipment.status = "blocked"
                shipment.failure_reason = "目标容器容量不足"
                self._upsert_shipment_tasks(shipment, now)
                self._event("shipment_blocked", order, now, shipment)
            else:
                resource.location = container.location_id
                resource.container_id = container.id
                shipment.status = "received"
                shipment.received_at = now
                order.status = "delivered"
                self._complete_tasks(shipment, now)
                self._event("shipment_received", order, now, shipment)

    @staticmethod
    def _travel_minutes(route) -> int:
        if not route:
            return 60
        values = [int(value) for value in route.travel_minutes.values() if int(value) > 0]
        return max(values) if values else 60

    def _upsert_shipment_tasks(self, shipment: Shipment, now: int) -> None:
        if shipment.status in {"planned", "dispatched"}:
            self.tasks.upsert(Task(
                id=f"logistics:dispatch:{shipment.id}", title="安排运输出发",
                assignee_id=shipment.carrier_id, source="logistics",
                location_id=shipment.source_location, interaction_type="wait",
                duration_minutes=5, earliest_at=now, priority=0.8,
                payload={"shipment_id": shipment.id}, created_at=now,
            ))
        elif shipment.status == "in_transit":
            self.tasks.complete(f"logistics:dispatch:{shipment.id}", now)
            self.tasks.upsert(Task(
                id=f"logistics:receive:{shipment.id}", title="接收运输批次",
                assignee_id=shipment.carrier_id, source="logistics",
                location_id=shipment.destination_location, interaction_type="wait",
                duration_minutes=5, earliest_at=shipment.arrival_at or now,
                priority=0.85, payload={"shipment_id": shipment.id}, created_at=now,
            ))
        elif shipment.status == "arrived":
            self.tasks.start(f"logistics:receive:{shipment.id}", now)
        elif shipment.status == "blocked":
            self.tasks.block(f"logistics:receive:{shipment.id}", shipment.failure_reason)

    def _complete_tasks(self, shipment: Shipment, now: int) -> None:
        self.tasks.complete(f"logistics:dispatch:{shipment.id}", now)
        self.tasks.complete(f"logistics:receive:{shipment.id}", now)

    def _fail_order(self, order: Order, reason: str) -> None:
        order.status = "failed"
        order.failure_reason = reason

    def _fail_shipment(self, order: Order, shipment: Shipment, reason: str) -> None:
        shipment.status = "failed"
        shipment.failure_reason = reason
        order.status = "failed"
        order.failure_reason = reason

    def _event(self, event_type: str, order: Order, now: int, shipment: Shipment | None = None) -> None:
        event = {"type": event_type, "order_id": order.id, "sim_timestamp": now}
        if shipment:
            event["shipment_id"] = shipment.id
        self.events.append(event)

    def state(self) -> dict:
        return {
            "orders": [order.to_dict() for order in self.orders.values()],
            "shipments": [shipment.to_dict() for shipment in self.shipments.values()],
            "events": list(self.events),
        }
