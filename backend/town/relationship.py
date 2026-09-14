"""Relationship system: asymmetric pairwise model with narrative anchors."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiosqlite
import os
from ..config import config


@dataclass
class Anchor:
    event: str
    valence: float  # -3 to +3
    time: str


@dataclass
class Relationship:
    agent_a: str
    agent_b: str
    familiarity: float = 0.0   # 0-10
    affinity: float = 0.0      # -10 to +10
    trust: float = 0.0         # 0-10
    anchors: list[Anchor] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction: str = ""
    last_interaction_id: str = ""

    def to_dict(self) -> dict:
        return {
            "agentA": self.agent_a, "agentB": self.agent_b,
            "familiarity": round(self.familiarity, 1),
            "affinity": round(self.affinity, 1),
            "trust": round(self.trust, 1),
            "anchors": [{"event": a.event, "valence": round(a.valence, 2), "time": a.time} for a in self.anchors],
            "tags": self.tags,
            "interactionCount": self.interaction_count,
            "lastInteraction": self.last_interaction,
            "lastInteractionId": self.last_interaction_id,
        }


class RelationshipStore:
    def __init__(self):
        self.db_path = config.DB_PATH

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    agent_a TEXT NOT NULL,
                    agent_b TEXT NOT NULL,
                    familiarity REAL DEFAULT 0,
                    affinity REAL DEFAULT 0,
                    trust REAL DEFAULT 0,
                    anchors TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TEXT DEFAULT '',
                    last_interaction_id TEXT DEFAULT '',
                    PRIMARY KEY (agent_a, agent_b)
                )
            """)
            try:
                await db.execute("ALTER TABLE relationships ADD COLUMN last_interaction_id TEXT DEFAULT ''")
            except aiosqlite.OperationalError:
                pass
            await db.commit()

    async def get(self, a: str, b: str) -> Relationship:
        """Get relationship from a's perspective toward b (asymmetric)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM relationships WHERE agent_a = ? AND agent_b = ?", (a, b)
            )
            row = await cursor.fetchone()
            if row:
                import json
                return Relationship(
                    agent_a=row[0], agent_b=row[1],
                    familiarity=row[2], affinity=row[3], trust=row[4],
                    anchors=[Anchor(**x) for x in json.loads(row[5])],
                    tags=json.loads(row[6]),
                    interaction_count=row[7], last_interaction=row[8],
                    last_interaction_id=row[9] if len(row) > 9 else "",
                )
            return Relationship(agent_a=a, agent_b=b)

    async def get_all_for(self, a: str) -> list[Relationship]:
        """Get all relationships from a's perspective."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM relationships WHERE agent_a = ? ORDER BY familiarity DESC", (a,)
            )
            rows = await cursor.fetchall()
        import json
        return [
            Relationship(
                agent_a=r[0], agent_b=r[1], familiarity=r[2], affinity=r[3], trust=r[4],
                anchors=[Anchor(**x) for x in json.loads(r[5])],
                tags=json.loads(r[6]), interaction_count=r[7], last_interaction=r[8],
                last_interaction_id=r[9] if len(r) > 9 else "",
            )
            for r in rows
        ]

    async def get_bidirectional(self, a: str, b: str) -> tuple[Relationship, Relationship]:
        """Get both directions of a relationship."""
        return await self.get(a, b), await self.get(b, a)

    async def upsert(self, rel: Relationship):
        import json
        rel.familiarity = round(max(0.0, min(10.0, rel.familiarity)), 2)
        rel.affinity = round(max(-10.0, min(10.0, rel.affinity)), 2)
        rel.trust = round(max(0.0, min(10.0, rel.trust)), 2)
        for anchor in rel.anchors:
            anchor.valence = round(max(-3.0, min(3.0, anchor.valence)), 2)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO relationships
                   (agent_a, agent_b, familiarity, affinity, trust, anchors, tags, interaction_count, last_interaction, last_interaction_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rel.agent_a, rel.agent_b, rel.familiarity, rel.affinity, rel.trust,
                 json.dumps([{"event": a.event, "valence": a.valence, "time": a.time} for a in rel.anchors], ensure_ascii=False),
                 json.dumps(rel.tags, ensure_ascii=False),
                 rel.interaction_count, rel.last_interaction, rel.last_interaction_id),
            )
            await db.commit()

    async def record_interaction(self, a: str, b: str, event: str, valence: float = 0.0,
                                  tag: str | None = None, affinity_delta: float = 0.0,
                                  trust_delta: float = 0.0,
                                  sim_time: str | None = None,
                                  interaction_id: str = ""):
        """Record an interaction, updating both directions."""
        rel_a = await self.get(a, b)
        rel_a.familiarity = round(min(10, rel_a.familiarity + config.RELATIONSHIP_FAMILIARITY_TALK_INCREMENT), 2)
        rel_a.affinity = round(max(-10, min(10, rel_a.affinity + affinity_delta)), 2)
        rel_a.trust = round(max(0, min(10, rel_a.trust + trust_delta)), 2)
        rel_a.interaction_count += 1
        rel_a.last_interaction = sim_time or datetime.now().isoformat()
        rel_a.last_interaction_id = interaction_id
        rel_a.anchors.append(Anchor(event=event, valence=valence, time=sim_time or datetime.now().isoformat()))
        if len(rel_a.anchors) > config.MAX_ANCHORS_PER_RELATIONSHIP:
            rel_a.anchors = rel_a.anchors[-config.MAX_ANCHORS_PER_RELATIONSHIP:]
        if tag and tag not in rel_a.tags:
            rel_a.tags.append(tag)

        rel_b = await self.get(b, a)
        rel_b.familiarity = round(min(10, rel_b.familiarity + config.RELATIONSHIP_FAMILIARITY_TALK_INCREMENT * 0.9), 2)
        rel_b.affinity = round(max(-10, min(10, rel_b.affinity + affinity_delta * 0.8)), 2)
        rel_b.trust = round(max(0, min(10, rel_b.trust + trust_delta * 0.8)), 2)
        rel_b.interaction_count += 1
        rel_b.last_interaction = sim_time or datetime.now().isoformat()
        rel_b.last_interaction_id = interaction_id

        rel_b.anchors.append(Anchor(
            event=event, valence=round(valence * 0.8, 2),
            time=sim_time or datetime.now().isoformat(),
        ))
        if len(rel_b.anchors) > config.MAX_ANCHORS_PER_RELATIONSHIP:
            rel_b.anchors = rel_b.anchors[-config.MAX_ANCHORS_PER_RELATIONSHIP:]

        await self.upsert(rel_a)
        await self.upsert(rel_b)

    async def record_colocation(self, agents: list[str]):
        """Being in the same location slowly increases familiarity."""
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                rel_a = await self.get(a, b)
                rel_a.familiarity = min(10, rel_a.familiarity + config.RELATIONSHIP_FAMILIARITY_COLOCATION_INCREMENT)
                await self.upsert(rel_a)
                rel_b = await self.get(b, a)
                rel_b.familiarity = min(10, rel_b.familiarity + config.RELATIONSHIP_FAMILIARITY_COLOCATION_INCREMENT)
                await self.upsert(rel_b)

    async def clear(self):
        """Delete all relationship data."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM relationships")
            await db.commit()

    async def get_behavior_modifier(self, a: str, b: str) -> dict:
        """Derive behavioral modifiers from relationship state.

        Returns a dict with keys:
          - talk_probability_bonus: float, additive modifier to dialogue probability
          - trust_level: float, 0.0–1.0
          - formality: float, 0.0–1.0 (higher = more formal)
          - help_willingness: float, 0.0–1.0

        Driven by anchors (valence), tags, affinity, and familiarity.
        """
        rel = await self.get(a, b)

        talk_probability_bonus = 0.0
        trust_level = 0.5
        formality = 0.5
        help_willingness = 0.5

        # ── Anchor influence: average valence drives talk + helpfulness ──
        if rel.anchors:
            avg_valence = sum(a.valence for a in rel.anchors) / len(rel.anchors)
            talk_probability_bonus += avg_valence * 0.05
            help_willingness += avg_valence * 0.05

        # ── Tag-driven modifiers ──
        for tag in rel.tags:
            t = tag.lower()
            if t in ("doctor", "医生"):
                trust_level = min(1.0, trust_level + 0.3)
                formality = min(1.0, formality + 0.15)
            elif t in ("neighbor", "邻居"):
                talk_probability_bonus += 0.15
            elif t in ("friend", "朋友"):
                talk_probability_bonus += 0.15
                help_willingness = min(1.0, help_willingness + 0.2)
            elif t in ("rival", "enemy", "敌人"):
                talk_probability_bonus -= 0.15
                help_willingness = max(0.0, help_willingness - 0.2)
                trust_level = max(0.0, trust_level - 0.2)

        # ── Affinity influence ──
        affinity_norm = rel.affinity / 10.0  # -1.0 … +1.0
        talk_probability_bonus += affinity_norm * 0.2
        help_willingness += affinity_norm * 0.2

        # Penalize low affinity — forced ceiling on talk bonus
        if rel.affinity < -3:
            talk_probability_bonus = min(talk_probability_bonus, -0.05)

        # ── Familiarity influence ──
        if rel.familiarity > 5:
            formality = max(0.0, formality - 0.2)
            talk_probability_bonus += 0.1
        elif rel.familiarity < 2:
            formality = min(1.0, formality + 0.15)

        # Clamp all outputs
        talk_probability_bonus = max(-0.5, min(0.5, talk_probability_bonus))
        trust_level = max(0.0, min(1.0, trust_level))
        formality = max(0.0, min(1.0, formality))
        help_willingness = max(0.0, min(1.0, help_willingness))

        return {
            "talk_probability_bonus": round(talk_probability_bonus, 3),
            "trust_level": round(trust_level, 3),
            "formality": round(formality, 3),
            "help_willingness": round(help_willingness, 3),
        }

    async def detect_relationship_arc(self, a: str, b: str) -> str | None:
        """Detect the current narrative arc of a relationship.

        Returns one of:
          - "warming up"    — interaction_count increased by 5+ in last 24 h
          - "cooling down"  — recent anchor valence suggests an affinity drop
          - "drifting apart" — no interaction in 72 h
          - None            — no detectable arc
        """
        rel = await self.get(a, b)
        now = datetime.now()
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_72h = (now - timedelta(hours=72)).isoformat()

        # Count interactions in last 24 h via anchor timestamps
        if rel.anchors:
            recent_count = sum(1 for a in rel.anchors if a.time >= cutoff_24h)
            if recent_count >= 5:
                return "warming up"

            # Sum of valences in last 24 h — a sharp negative net → cooling
            recent_valence_sum = sum(a.valence for a in rel.anchors if a.time >= cutoff_24h)
            if recent_valence_sum <= -3:
                return "cooling down"

        # Drifting: no interaction recorded, or last interaction > 72 h ago
        if not rel.last_interaction:
            return "drifting apart"
        try:
            last = datetime.fromisoformat(rel.last_interaction)
            if (now - last) > timedelta(hours=72):
                return "drifting apart"
        except (ValueError, TypeError):
            pass

        return None

    def format_for_llm(self, rel: Relationship, their_name: str) -> str:
        """Format a relationship for LLM context."""
        if rel.familiarity < 0.5:
            return f"你和{their_name}还不熟悉。"
        parts = [f"你和{their_name}的关系："]
        if rel.familiarity > 5:
            parts.append(f"- 你们很熟悉（熟悉度 {rel.familiarity:.0f}/10）")
        elif rel.familiarity > 2:
            parts.append(f"- 你们还算认识（熟悉度 {rel.familiarity:.0f}/10）")
        if rel.affinity > 3:
            parts.append(f"- 你对他/她印象很好")
        elif rel.affinity < -3:
            parts.append(f"- 你对他/她有些不满")
        if rel.tags:
            parts.append(f"- 关系标签：{', '.join(rel.tags)}")
        if rel.anchors:
            recent = rel.anchors[-3:]
            parts.append("- 共同的经历：")
            for a in recent:
                parts.append(f"  · {a.event}")
        return "\n".join(parts)
