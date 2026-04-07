import fs from "fs";
import path from "path";

export type StationNode = {
  id: string;
  type: "Station";
  name: string;
  city: string;
  province: string;
  river: string;
  lon: number;
  lat: number;
  order: number;
  is_key: boolean;
  data_count: number;
  data_period: string;
  latest_do: number;
  main_quality_class: string;
};

export type ModelResultNode = {
  id: string;
  type: "ModelResult";
  station: string;
  rmse: number;
  mae: number;
  r2: number;
  mape: number;
  top_features?: Array<{ name: string; importance: number; explanation?: string }>;
};

export type ForecastPoint = {
  hours_ahead: number;
  time: string;
  time_full: string;
  do: number;
  delta: number;
  trend: string;
  quality: string;
  status: string;
  color: string;
};

export type StationForecast = {
  current_do: number;
  current_time: string;
  current_quality: string;
  current_status: string;
  current_color: string;
  forecasts: ForecastPoint[];
  recent_history: Array<{ time: string; do: number }>;
};

type KgData = {
  metadata: {
    project: string;
    total_stations: number;
    key_stations: number;
    rivers: string[];
    model: string;
    target: string;
    forecast_horizon: string;
  };
  nodes: Array<StationNode | ModelResultNode | { type: string; [key: string]: unknown }>;
};

export type DashboardData = {
  metadata: KgData["metadata"];
  stations: StationNode[];
  modelResults: Record<string, ModelResultNode>;
  forecasts: Record<string, StationForecast>;
};

function readJsonFile<T>(relativePath: string): T {
  const repoRoot = path.resolve(process.cwd(), "..", "..");
  const filePath = path.join(repoRoot, relativePath);
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw) as T;
}

export function getDashboardData(): DashboardData {
  const kg = readJsonFile<KgData>("src/knowledge_graph/kg_data.json");
  const forecasts = readJsonFile<Record<string, StationForecast>>("models/forecasts.json");

  const stations = kg.nodes
    .filter((node): node is StationNode => node.type === "Station")
    .sort((a, b) => (a.river === b.river ? a.order - b.order : a.river.localeCompare(b.river)));

  const modelResults = kg.nodes
    .filter((node): node is ModelResultNode => node.type === "ModelResult")
    .reduce<Record<string, ModelResultNode>>((acc, item) => {
      acc[item.station] = item;
      return acc;
    }, {});

  return {
    metadata: kg.metadata,
    stations,
    modelResults,
    forecasts,
  };
}

export function normalizeStationName(name: string): string {
  return decodeURIComponent(name ?? "");
}

export function getStationDetail(name: string) {
  const data = getDashboardData();
  const stationName = normalizeStationName(name);
  const station = data.stations.find((item) => item.name === stationName);
  if (!station) {
    return null;
  }

  return {
    station,
    forecast: data.forecasts[stationName],
    model: data.modelResults[stationName],
    metadata: data.metadata,
  };
}

export function getTopStationsByR2(limit = 7) {
  const data = getDashboardData();
  return Object.values(data.modelResults)
    .sort((a, b) => b.r2 - a.r2)
    .slice(0, limit);
}
