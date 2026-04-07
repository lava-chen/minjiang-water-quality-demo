"use client";

import L, { type DivIcon } from "leaflet";
import {
  GeoJSON,
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  Tooltip as LeafletTooltip,
} from "react-leaflet";
import MarkerClusterGroup from "react-leaflet-cluster";

import type { StationNode } from "@/lib/data";
import { qualityLabel } from "@/lib/quality";

type StationMapProps = {
  stations: StationNode[];
  activeStation?: string;
  onSelectStation?: (stationName: string) => void;
};

const riverColorPalette = [
  "#0ea5e9",
  "#06b6d4",
  "#14b8a6",
  "#22c55e",
  "#84cc16",
  "#f59e0b",
  "#f97316",
  "#ef4444",
];

function getCenter(stations: StationNode[]): [number, number] {
  const lats = stations.map((s) => s.lat);
  const lons = stations.map((s) => s.lon);
  const lat = lats.reduce((sum, item) => sum + item, 0) / Math.max(lats.length, 1);
  const lon = lons.reduce((sum, item) => sum + item, 0) / Math.max(lons.length, 1);
  return [lat || 30.3, lon || 103.9];
}

function getMarkerClass(quality: string) {
  if (quality === "I") return "marker-I";
  if (quality === "II") return "marker-II";
  if (quality === "III") return "marker-III";
  if (quality === "IV") return "marker-IV";
  return "marker-V";
}

function createStationIcon(quality: string, isActive: boolean): DivIcon {
  return L.divIcon({
    className: "",
    html: `<div class="station-marker ${getMarkerClass(quality)} ${isActive ? "active" : ""}"></div>`,
    iconSize: isActive ? [20, 20] : [16, 16],
    iconAnchor: isActive ? [10, 10] : [8, 8],
    popupAnchor: [0, -10],
  });
}

function buildRiverGeoJson(stations: StationNode[]) {
  const riverMap = new Map<string, StationNode[]>();
  stations.forEach((station) => {
    const list = riverMap.get(station.river) ?? [];
    list.push(station);
    riverMap.set(station.river, list);
  });

  const riverNames = [...riverMap.keys()].sort((a, b) => a.localeCompare(b));
  const colorMap = new Map<string, string>();
  riverNames.forEach((river, idx) => {
    colorMap.set(river, riverColorPalette[idx % riverColorPalette.length]);
  });

  const features = [...riverMap.entries()]
    .map(([river, list]) => {
      const ordered = [...list].sort((a, b) => a.order - b.order);
      if (ordered.length < 2) {
        return null;
      }

      return {
        type: "Feature" as const,
        properties: {
          river,
          color: colorMap.get(river) ?? "#0ea5e9",
        },
        geometry: {
          type: "LineString" as const,
          coordinates: ordered.map((station) => [station.lon, station.lat]),
        },
      };
    })
    .filter(Boolean);

  return {
    featureCollection: {
      type: "FeatureCollection" as const,
      features,
    },
    colorMap,
  };
}

export function StationMap({ stations, activeStation, onSelectStation }: StationMapProps) {
  const center = getCenter(stations);
  const { featureCollection, colorMap } = buildRiverGeoJson(stations);

  return (
    <div className="relative h-full w-full">
      <MapContainer center={center} zoom={7} scrollWheelZoom className="h-full w-full rounded-xl">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />

        <GeoJSON
          data={featureCollection}
          style={(feature) => ({
            color: String(feature?.properties?.color ?? "#0ea5e9"),
            weight: 3,
            opacity: 0.72,
          })}
          onEachFeature={(feature, layer) => {
            const riverName = String(feature.properties?.river ?? "");
            layer.bindTooltip(riverName, {
              sticky: true,
              direction: "top",
            });
          }}
        />

        <MarkerClusterGroup
          chunkedLoading
          spiderfyOnMaxZoom
          showCoverageOnHover={false}
          maxClusterRadius={40}
        >
          {stations.map((station) => {
            const isActive = station.name === activeStation;
            return (
              <Marker
                key={station.id}
                position={[station.lat, station.lon]}
                icon={createStationIcon(station.main_quality_class, isActive)}
                eventHandlers={{
                  click: () => onSelectStation?.(station.name),
                }}
              >
                <Popup>
                  <div className="text-sm">
                    <div className="font-semibold">{station.name}</div>
                    <div>{station.river}</div>
                    <div>{station.city}</div>
                    <div>DO: {station.latest_do.toFixed(2)} mg/L</div>
                    <div>{qualityLabel(station.main_quality_class)}</div>
                    <div className="mt-1 text-xs text-slate-500">{station.is_key ? "重点站" : "普通站"}</div>
                  </div>
                </Popup>
                <LeafletTooltip direction="top" offset={[0, -10]} opacity={0.9}>
                  {station.name}
                </LeafletTooltip>
              </Marker>
            );
          })}
        </MarkerClusterGroup>
      </MapContainer>

      <div className="pointer-events-none absolute bottom-3 left-3 z-[500] rounded-xl border border-white/80 bg-white/90 p-3 text-[11px] text-slate-700 shadow-md backdrop-blur">
        <div className="mb-1 font-semibold text-slate-900">水质等级</div>
        <div className="grid grid-cols-3 gap-x-3 gap-y-1">
          <LegendDot cls="marker-I" label="I类" />
          <LegendDot cls="marker-II" label="II类" />
          <LegendDot cls="marker-III" label="III类" />
          <LegendDot cls="marker-IV" label="IV类" />
          <LegendDot cls="marker-V" label="V类" />
        </div>
        <div className="mb-1 mt-2 font-semibold text-slate-900">河流线</div>
        <div className="grid max-h-28 grid-cols-2 gap-x-3 gap-y-1 overflow-auto pr-1">
          {[...colorMap.entries()].map(([river, color]) => (
            <div key={river} className="flex items-center gap-1.5 whitespace-nowrap">
              <span className="river-legend-line" style={{ borderColor: color }} />
              <span>{river}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LegendDot({ cls, label }: { cls: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`station-marker ${cls}`} />
      <span>{label}</span>
    </div>
  );
}
