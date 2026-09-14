import { useCallback, useEffect, useRef } from 'react';
import {
  Application,
  Assets,
  Container,
  Graphics,
  Rectangle,
  Sprite,
  Text,
  Texture,
} from 'pixi.js';
import { AgentState, Event, Location, WorldState } from '../types';

interface TownMapProps {
  locations: Location[];
  agents: AgentState[];
  selectedAgentId: string | null;
  recentEvents: Event[];
  worldState: WorldState;
  onSelectAgent: (id: string | null) => void;
}

interface TownTextures {
  urban: Texture;
  roguelike: Texture;
}

const TILE_PIXELS = 16;
const TILE_STRIDE = 17;
const URBAN_ASSET = '/assets/town/kenney-rpg-urban.png';
const ROGUELIKE_ASSET = '/assets/town/kenney-roguelike-rpg.png';

const WEATHER_EMOJI: Record<string, string> = {
  '晴': '\u2600\uFE0F', '多云': '\u26C5', '阴': '\u2601\uFE0F',
  '小雨': '\uD83C\uDF27\uFE0F', '大雨': '\uD83C\uDF27\uFE0F', '暴雨': '\u26C8\uFE0F',
};

interface BuildingStyle {
  roof: number;
  wall: number;
  trim: number;
  awning: number;
  window: number;
}

const BUILDING_STYLES: Record<string, BuildingStyle> = {
  home: { roof: 0xa95345, wall: 0xf0d4a0, trim: 0x6c4439, awning: 0xb55d46, window: 0x9bd4db },
  cafe: { roof: 0x916144, wall: 0xf2d19a, trim: 0x694535, awning: 0x4e9d91, window: 0xaee2e3 },
  shop: { roof: 0x4f718d, wall: 0xd7e1d7, trim: 0x3d5361, awning: 0xd87854, window: 0xb9e5e5 },
  restaurant: { roof: 0xb74e48, wall: 0xf0c49a, trim: 0x704039, awning: 0xd67243, window: 0xbde4db },
  clinic: { roof: 0x5b7890, wall: 0xd9e8df, trim: 0x476172, awning: 0x5e9f92, window: 0xc1e7ea },
  school: { roof: 0x6373a5, wall: 0xe6d8ad, trim: 0x4c557c, awning: 0xc99642, window: 0xb9dfdf },
  market: { roof: 0x698648, wall: 0xe8d199, trim: 0x526c3c, awning: 0x78a75b, window: 0xb4e0d8 },
  community: { roof: 0x88705b, wall: 0xe2d7bb, trim: 0x67574a, awning: 0xbd8c5c, window: 0xbadfe2 },
  pharmacy: { roof: 0x557c7a, wall: 0xe0ead5, trim: 0x3d615f, awning: 0x66a890, window: 0xc3e8e4 },
  transit: { roof: 0x566d86, wall: 0xc7d8db, trim: 0x3e5267, awning: 0x6689a7, window: 0xbfe6e8 },
};

const DEFAULT_BUILDING_STYLE: BuildingStyle = {
  roof: 0x6f6b63, wall: 0xe1d4b9, trim: 0x554f49, awning: 0xa77b50, window: 0xb8dfe0,
};

const AGENT_SPRITES: Record<string, [number, number]> = {
  wang: [23, 9],
  mei: [23, 4],
  liu: [23, 0],
  hua: [23, 12],
  ming: [23, 15],
  li: [24, 9],
};

function makeFrame(sheet: Texture, column: number, row: number): Texture {
  return new Texture({
    source: sheet.source,
    frame: new Rectangle(column * TILE_STRIDE, row * TILE_STRIDE, TILE_PIXELS, TILE_PIXELS),
  });
}

function makeLabel(text: string, size: number, fill = 0xf5f1df): Text {
  return new Text({
    text,
    style: {
      fill,
      fontFamily: 'Arial, PingFang SC, Microsoft YaHei, sans-serif',
      fontSize: size,
      fontWeight: '600',
      stroke: { color: 0x1e2a30, width: Math.max(1, size * 0.12) },
    },
  });
}

function destroyStage(stage: Container): void {
  const children = stage.removeChildren();
  children.forEach(child => child.destroy({ children: true }));
}

export default function TownMap({
  locations,
  agents,
  selectedAgentId,
  recentEvents,
  worldState,
  onSelectAgent,
}: TownMapProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<Application | null>(null);
  const texturesRef = useRef<TownTextures | null>(null);
  const renderRef = useRef<() => void>(() => {});

  const renderMap = useCallback(() => {
    const app = appRef.current;
    const textures = texturesRef.current;
    if (!app || !textures || app.renderer.width < 2 || app.renderer.height < 2) return;

    const gridWidth = worldState.map?.width || Math.max(1, ...locations.map(location => location.x + location.width));
    const gridHeight = worldState.map?.height || Math.max(1, ...locations.map(location => location.y + location.height));
    const availableCell = Math.min(app.renderer.width / gridWidth, app.renderer.height / gridHeight);
    // Four-pixel increments keep sprites crisp while letting the map use the full viewport.
    const cell = Math.max(TILE_PIXELS, Math.floor(availableCell / 4) * 4);
    const mapWidth = gridWidth * cell;
    const mapHeight = gridHeight * cell;
    const offsetX = Math.round((app.renderer.width - mapWidth) / 2);
    const offsetY = Math.round((app.renderer.height - mapHeight) / 2);
    const scale = cell / TILE_PIXELS;
    const urbanFrame = (column: number, row: number) => makeFrame(textures.urban, column, row);

    destroyStage(app.stage);
    app.stage.sortableChildren = true;

    const background = new Graphics()
      .rect(0, 0, app.renderer.width, app.renderer.height)
      .fill({ color: 0x172326 });
    background.zIndex = -2;
    app.stage.addChild(background);

    const map = new Container();
    map.position.set(offsetX, offsetY);
    map.zIndex = 0;
    app.stage.addChild(map);

    // Tile (1, 1) is the uninterrupted lawn interior; edge tiles create false grid borders.
    for (let y = 0; y < gridHeight; y++) {
      for (let x = 0; x < gridWidth; x++) {
        const ground = new Sprite(urbanFrame(1, 1));
        ground.position.set(x * cell, y * cell);
        ground.scale.set(scale);
        map.addChild(ground);
      }
    }

    const roadCells = new Set<string>();
    locations.filter(location => location.type === 'road').forEach(location => {
      for (let y = location.y; y < location.y + location.height; y++) {
        for (let x = location.x; x < location.x + location.width; x++) roadCells.add(`${x},${y}`);
      }
    });
    roadCells.forEach(cellId => {
      const [gridX, gridY] = cellId.split(',').map(Number);
      const x = gridX * cell;
      const y = gridY * cell;
      const north = roadCells.has(`${gridX},${gridY - 1}`);
      const east = roadCells.has(`${gridX + 1},${gridY}`);
      const south = roadCells.has(`${gridX},${gridY + 1}`);
      const west = roadCells.has(`${gridX - 1},${gridY}`);
      const road = new Graphics().rect(x, y, cell, cell).fill({ color: 0x46555b });
      const curbWidth = Math.max(1, cell * 0.09);
      if (!north) road.rect(x, y, cell, curbWidth).fill({ color: 0xd7c99f });
      if (!east) road.rect(x + cell - curbWidth, y, curbWidth, cell).fill({ color: 0xd7c99f });
      if (!south) road.rect(x, y + cell - curbWidth, cell, curbWidth).fill({ color: 0xd7c99f });
      if (!west) road.rect(x, y, curbWidth, cell).fill({ color: 0xd7c99f });
      if (east || west) road.rect(x + cell * 0.24, y + cell * 0.46, cell * 0.52, Math.max(1, cell * 0.07)).fill({ color: 0xf4e3a6 });
      if (north || south) road.rect(x + cell * 0.46, y + cell * 0.24, Math.max(1, cell * 0.07), cell * 0.52).fill({ color: 0xf4e3a6 });
      map.addChild(road);
    });

    const park = locations.find(location => location.type === 'park');
    if (park) {
      const parkBase = new Graphics()
        .rect(park.x * cell + cell * 0.1, park.y * cell + cell * 0.1,
          park.width * cell - cell * 0.2, park.height * cell - cell * 0.2)
        .fill({ color: 0x4d8d54 })
        .stroke({ color: 0x2f6948, width: Math.max(1, cell * 0.06) });
      map.addChild(parkBase);
      const path = new Graphics()
        .rect((park.x + 0.42) * cell, (park.y + 1.82) * cell, (park.width - 0.84) * cell, cell * 0.28)
        .fill({ color: 0xdccf9c, alpha: 0.88 });
      map.addChild(path);
      const trees: [number, number][] = [[0.45, 0.28], [2.25, 0.46], [5.2, 0.3], [1.3, 2.18], [4.4, 2.1]];
      trees.forEach(([x, y], index) => {
        const tree = new Sprite(urbanFrame(index % 2 === 0 ? 16 : 17, 9));
        tree.anchor.set(0.5, 1);
        tree.position.set((park.x + x) * cell, (park.y + y + 0.96) * cell);
        tree.scale.set(scale * 1.38);
        map.addChild(tree);
      });
    }

    const recentLocationIds = new Set(recentEvents.slice(-8).map(event => event.location));
    for (const location of locations) {
      if (location.type === 'road' || location.type === 'park') continue;
      const x = location.x * cell;
      const y = location.y * cell;
      const width = location.width * cell;
      const height = location.height * cell;
      const style = BUILDING_STYLES[location.type] || DEFAULT_BUILDING_STYLE;
      const inset = Math.max(2, cell * 0.1);
      const roofHeight = Math.min(height * 0.44, cell * 0.98);
      const facadeY = y + roofHeight * 0.72;
      const facadeHeight = height - roofHeight * 0.72 - inset;
      const building = new Graphics();

      building.rect(x + inset, y + inset + cell * 0.12, width - inset, height - inset).fill({ color: 0x172426, alpha: 0.48 });
      building.rect(x + inset, facadeY, width - inset * 2, facadeHeight).fill({ color: style.wall });
      building.rect(x + inset, facadeY, width - inset * 2, facadeHeight).stroke({ color: style.trim, width: Math.max(1, cell * 0.06) });
      building.poly([
        x + inset, facadeY + cell * 0.08,
        x + width * 0.5, y + inset,
        x + width - inset, facadeY + cell * 0.08,
      ], true).fill({ color: style.roof }).stroke({ color: style.trim, width: Math.max(1, cell * 0.06) });
      building.rect(x + inset * 1.4, facadeY - cell * 0.03, width - inset * 2.8, cell * 0.1).fill({ color: style.trim, alpha: 0.7 });

      const windowWidth = Math.max(cell * 0.22, Math.min(cell * 0.48, width * 0.16));
      const windowHeight = Math.max(cell * 0.2, facadeHeight * 0.27);
      const windowY = facadeY + facadeHeight * 0.5;
      building.rect(x + width * 0.19 - windowWidth / 2, windowY, windowWidth, windowHeight).fill({ color: style.window }).stroke({ color: style.trim, width: Math.max(1, cell * 0.04) });
      building.rect(x + width * 0.81 - windowWidth / 2, windowY, windowWidth, windowHeight).fill({ color: style.window }).stroke({ color: style.trim, width: Math.max(1, cell * 0.04) });
      if (location.width >= 4) {
        building.rect(x + width * 0.34 - windowWidth / 2, windowY, windowWidth, windowHeight).fill({ color: style.window }).stroke({ color: style.trim, width: Math.max(1, cell * 0.04) });
        building.rect(x + width * 0.66 - windowWidth / 2, windowY, windowWidth, windowHeight).fill({ color: style.window }).stroke({ color: style.trim, width: Math.max(1, cell * 0.04) });
      }
      const doorWidth = Math.max(cell * 0.27, width * 0.13);
      const doorHeight = Math.max(cell * 0.36, facadeHeight * 0.44);
      building.rect(x + width / 2 - doorWidth / 2, y + height - inset - doorHeight, doorWidth, doorHeight).fill({ color: style.trim });
      building.rect(x + width / 2 + doorWidth * 0.22, y + height - inset - doorHeight * 0.48, Math.max(1, cell * 0.04), Math.max(1, cell * 0.04)).fill({ color: 0xf7d47c });
      map.addChild(building);

      const label = makeLabel(location.name, Math.max(8, Math.min(cell * 0.2, width / Math.max(4, location.name.length * 2.8))));
      const signWidth = Math.min(width - cell * 0.3, label.width + cell * 0.36);
      const sign = new Graphics()
        .rect(x + width / 2 - signWidth / 2, facadeY + cell * 0.14, signWidth, Math.max(cell * 0.23, label.height + cell * 0.1))
        .fill({ color: style.awning })
        .stroke({ color: style.trim, width: Math.max(1, cell * 0.04) });
      map.addChild(sign);
      label.anchor.set(0.5, 0.5);
      label.position.set(x + width / 2, facadeY + cell * 0.14 + Math.max(cell * 0.23, label.height + cell * 0.1) / 2);
      map.addChild(label);

      const people = agents.filter(agent => agent.currentLocation === location.id).length;
      if (people > 0) {
        const occupancy = new Graphics()
          .roundRect(x + width - cell * 0.48, y + height - cell * 0.46, cell * 0.34, cell * 0.28, cell * 0.08)
          .fill({ color: 0x1d2a2a, alpha: 0.88 });
        const count = makeLabel(String(people), Math.max(8, cell * 0.17), 0xeaf5cf);
        count.anchor.set(0.5, 0.5);
        count.position.set(x + width - cell * 0.31, y + height - cell * 0.32);
        map.addChild(occupancy, count);
      }

      if (recentLocationIds.has(location.id)) {
        const pulse = new Graphics()
          .circle(x + width - cell * 0.16, y + cell * 0.16, cell * 0.13)
          .fill({ color: 0xffd76c })
          .stroke({ color: 0xfff7cf, width: Math.max(1, cell * 0.03) });
        map.addChild(pulse);
      }
    }

    const dialogueAgents = new Set(
      recentEvents.filter(event => event.type === 'dialogue_line').slice(-3).flatMap(event => event.agentIds),
    );
    agents.slice().sort((a, b) => a.y - b.y).forEach((agent, index) => {
      const person = new Container();
      const px = (agent.x + 0.5) * cell;
      const py = (agent.y + 0.82) * cell;
      person.position.set(px, py);
      person.zIndex = 100 + agent.y;
      person.eventMode = 'static';
      person.hitArea = new Rectangle(-cell * 0.42, -cell * 1.28, cell * 0.84, cell * 1.36);
      person.cursor = 'pointer';
      person.on('pointertap', () => onSelectAgent(agent.id === selectedAgentId ? null : agent.id));

      const shadow = new Graphics()
        .ellipse(0, 0, cell * 0.26, cell * 0.1)
        .fill({ color: 0x1a2424, alpha: 0.48 });
      person.addChild(shadow);

      if (agent.id === selectedAgentId) {
        const selection = new Graphics()
          .circle(0, -cell * 0.3, cell * 0.48)
          .stroke({ color: 0xffda67, width: Math.max(2, cell * 0.08), alpha: 0.96 });
        person.addChild(selection);
      }

      const [avatarColumn, avatarRow] = AGENT_SPRITES[agent.id] || [23 + (index % 4), 0];
      const avatar = new Sprite(urbanFrame(avatarColumn, avatarRow));
      avatar.anchor.set(0.5, 1);
      avatar.position.set(0, cell * 0.05);
      avatar.scale.set(scale * 1.15);
      person.addChild(avatar);

      const stateColor = agent.status === 'MOVING' ? 0x61b7e6
        : agent.status === 'ACTING' ? 0xf0b75b
          : agent.status === 'SPEAKING' || agent.status === 'LISTENING' ? 0xce80df : 0x8ed19d;
      const status = new Graphics()
        .roundRect(-cell * 0.34, cell * 0.12, cell * 0.68, cell * 0.1, cell * 0.05)
        .fill({ color: stateColor });
      person.addChild(status);

      const name = makeLabel(agent.name, Math.max(10, cell * 0.23));
      name.anchor.set(0.5, 1);
      name.position.set(0, -cell * 0.84);
      person.addChild(name);

      if (dialogueAgents.has(agent.id)) {
        const bubble = makeLabel('\uD83D\uDCAC', Math.max(12, cell * 0.4));
        bubble.anchor.set(0.5, 0.5);
        bubble.position.set(cell * 0.48, -cell * 0.92);
        person.addChild(bubble);
      }
      map.addChild(person);
    });
  }, [agents, locations, onSelectAgent, recentEvents, selectedAgentId, worldState.environment, worldState.map]);

  renderRef.current = renderMap;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    let resizeObserver: ResizeObserver | null = null;
    const app = new Application();

    const boot = async () => {
      await app.init({
        width: Math.max(1, host.clientWidth),
        height: Math.max(1, host.clientHeight),
        backgroundAlpha: 0,
        antialias: false,
        autoDensity: true,
        preference: 'webgl',
        resolution: Math.min(window.devicePixelRatio || 1, 2),
      });
      if (disposed) {
        app.destroy({ removeView: true }, { children: true });
        return;
      }
      appRef.current = app;
      host.replaceChildren(app.canvas);

      const [urban, roguelike] = await Promise.all([
        Assets.load<Texture>(URBAN_ASSET),
        Assets.load<Texture>(ROGUELIKE_ASSET),
      ]);
      if (disposed) return;
      urban.source.scaleMode = 'nearest';
      roguelike.source.scaleMode = 'nearest';
      texturesRef.current = { urban, roguelike };

      resizeObserver = new ResizeObserver(entries => {
        const rect = entries[0]?.contentRect;
        if (!rect || rect.width < 2 || rect.height < 2) return;
        app.renderer.resize(Math.floor(rect.width), Math.floor(rect.height));
        renderRef.current();
      });
      resizeObserver.observe(host);
      renderRef.current();
    };

    void boot();
    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      texturesRef.current = null;
      if (appRef.current === app) appRef.current = null;
      if (app.renderer) app.destroy({ removeView: true }, { children: true });
    };
  }, []);

  useEffect(() => {
    renderMap();
  }, [renderMap]);

  const weather = worldState.weather;
  const infrastructure = worldState.infrastructure;
  return (
    <div className="town-map-wrapper">
      <div className="weather-banner">
        <span className="weather-emoji">{WEATHER_EMOJI[weather.condition] || '\u2600\uFE0F'}</span>
        <span className="weather-temp">{weather.temperature}{'\u00B0'}C</span>
        <span className="weather-wind">{'\u{1F4A8}'} {weather.wind}</span>
        <span className="weather-desc">{weather.description}</span>
        <span className="infra-badges">
          {Object.entries(infrastructure).map(([service, status]) => (
            <span key={service} className={`infra-badge ${status}`}>{service}: {status}</span>
          ))}
        </span>
      </div>
      <div ref={hostRef} className="town-map-container town-map-pixi" aria-label="404小镇地图" />
      {worldState.townEvents.length > 0 && (
        <div className="town-events">
          {worldState.townEvents.map((event, index) => (
            <div key={index} className="town-event-item">
              <span className="te-desc">{event.description}</span>
              <span className="te-location">@{event.location}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
