"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Activity, Gauge, MapPinned, Waves } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AppShell } from "@/components/dashboard/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { DashboardData } from "@/lib/data";
import { qualityBadgeColor, qualityLabel } from "@/lib/quality";

const StationMap = dynamic(
  () => import("@/components/dashboard/station-map").then((m) => m.StationMap),
  { ssr: false, loading: () => <div className="h-full rounded-xl bg-cyan-50" /> },
);

type DashboardClientProps = {
  data: DashboardData;
};

const numberFmt = new Intl.NumberFormat("zh-CN");

export function DashboardClient({ data }: DashboardClientProps) {
  const keyStations = data.stations.filter((station) => station.is_key);
  const [activeStation, setActiveStation] = useState<string>(keyStations[0]?.name ?? data.stations[0]?.name ?? "");

  const summary = useMemo(() => {
    const avgR2 =
      Object.values(data.modelResults).reduce((sum, item) => sum + (item.r2 ?? 0), 0) /
      Math.max(Object.keys(data.modelResults).length, 1);

    return {
      stationCount: data.stations.length,
      keyStationCount: keyStations.length,
      riverCount: new Set(data.stations.map((station) => station.river)).size,
      avgR2,
    };
  }, [data.modelResults, data.stations, keyStations.length]);

  const activeForecast = data.forecasts[activeStation];
  const activeModel = data.modelResults[activeStation];
  const activeStationNode = data.stations.find((station) => station.name === activeStation);

  const trendData = useMemo(() => {
    if (!activeForecast) {
      return [] as Array<{ time: string; actual?: number; forecast?: number }>;
    }

    const history = activeForecast.recent_history.map((item) => ({
      time: item.time.slice(5),
      actual: item.do,
      forecast: undefined,
    }));

    const forecast = [
      {
        time: activeForecast.current_time.slice(5),
        actual: activeForecast.current_do,
        forecast: activeForecast.current_do,
      },
      ...activeForecast.forecasts.map((item) => ({
        time: item.time,
        actual: undefined,
        forecast: item.do,
      })),
    ];

    return [...history, ...forecast];
  }, [activeForecast]);

  const topFeatures =
    activeModel?.top_features?.slice(0, 6).map((item) => ({
      name: item.name,
      value: Number(item.importance.toFixed(4)),
    })) ?? [];

  return (
    <AppShell
      title="岷江流域水质智能数据看板"
      subtitle={`基于 ${data.metadata.model} 的多站点溶解氧预测，结合空间分布、趋势变化与驱动因子，辅助流域监测与预警。`}
    >
      <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={MapPinned} title="监测站点" value={String(summary.stationCount)} helper="全流域站点总数" />
        <MetricCard icon={Gauge} title="重点建模站" value={String(summary.keyStationCount)} helper="用于核心预测分析" />
        <MetricCard icon={Waves} title="覆盖河流" value={String(summary.riverCount)} helper="干流与主要支流" />
        <MetricCard icon={Activity} title="平均 R²" value={summary.avgR2.toFixed(3)} helper="预测拟合优度" />
      </section>

      <section className="mt-6 grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="border-white/60 bg-white/85 backdrop-blur">
          <CardHeader>
            <CardTitle>站点空间分布地图</CardTitle>
            <CardDescription>Carto 浅色地形底图（点击站点可切换右侧数据）</CardDescription>
          </CardHeader>
          <CardContent className="h-[360px]">
            <StationMap
              stations={data.stations}
              activeStation={activeStation}
              onSelectStation={(stationName) => setActiveStation(stationName)}
            />
          </CardContent>
        </Card>

        <Card className="border-white/60 bg-white/85 backdrop-blur">
          <CardHeader>
            <CardTitle>站点快照</CardTitle>
            <CardDescription>当前选中站点状态与 24h 预测</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-xl bg-gradient-to-r from-cyan-900 to-sky-800 p-4 text-white">
              <div className="text-sm text-cyan-100">
                {activeStationNode?.river} · {activeStationNode?.city}
              </div>
              <div className="mt-2 text-3xl font-semibold">{activeForecast?.current_do?.toFixed(2) ?? "--"} mg/L</div>
              <div className="mt-2 flex items-center gap-2">
                <Badge className={qualityBadgeColor(activeForecast?.current_quality ?? "V")}>
                  {qualityLabel(activeForecast?.current_quality ?? "-")}
                </Badge>
                <span className="text-sm text-cyan-100">{activeForecast?.current_status ?? "暂无状态"}</span>
              </div>
              <div className="mt-3 text-xs text-cyan-100">更新时间: {activeForecast?.current_time ?? "--"}</div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {activeForecast?.forecasts?.slice(0, 6).map((point) => (
                <div key={point.hours_ahead} className="rounded-xl border border-cyan-100 bg-cyan-50/70 p-3">
                  <div className="text-xs text-cyan-900">+{point.hours_ahead}h</div>
                  <div className="mt-1 text-lg font-semibold text-cyan-950">{point.do.toFixed(2)}</div>
                  <div className="text-xs text-cyan-700">{point.time}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="mt-6 grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="border-white/60 bg-white/85 backdrop-blur">
          <CardHeader>
            <CardTitle>溶解氧趋势</CardTitle>
            <CardDescription>近 7 天实测 + 未来 24h 预测（{activeStation}）</CardDescription>
          </CardHeader>
          <CardContent className="h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#bae6fd" />
                <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 11 }} unit=" mg/L" domain={["dataMin - 0.5", "dataMax + 0.5"]} />
                <Tooltip />
                <Area type="monotone" dataKey="actual" stroke="#0284c7" fillOpacity={1} fill="url(#actualFill)" name="实测" />
                <Area type="monotone" dataKey="forecast" stroke="#0f766e" fillOpacity={1} fill="url(#forecastFill)" name="预测" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>驱动因子 Top 6</CardTitle>
            <CardDescription>特征重要性（{activeStation}）</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 22, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbeafe" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#0891b2" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      <section className="mt-6 grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>重点站快速切换</CardTitle>
            <CardDescription>查看不同站点预测表现与详情页面</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {keyStations.map((station) => {
              const isActive = station.name === activeStation;
              return (
                <Button
                  key={station.id}
                  variant={isActive ? "default" : "outline"}
                  className={isActive ? "bg-cyan-700 hover:bg-cyan-600" : "border-cyan-200"}
                  onClick={() => setActiveStation(station.name)}
                  asChild
                >
                  <Link href={`/stations/${encodeURIComponent(station.name)}`}>{station.name}</Link>
                </Button>
              );
            })}
          </CardContent>
        </Card>

        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>快速操作</CardTitle>
            <CardDescription>进入解释页查看全局模型质量与驱动信息</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button className="w-full bg-cyan-700 hover:bg-cyan-600" asChild>
              <Link href="/insights">进入解释分析页</Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      <section className="mt-6">
        <Card className="border-white/60 bg-white/95">
          <CardHeader>
            <CardTitle>站点状态总览</CardTitle>
            <CardDescription>全站点基础信息、最新 DO 与模型表现</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>站点</TableHead>
                  <TableHead>河流</TableHead>
                  <TableHead>城市</TableHead>
                  <TableHead>最新 DO</TableHead>
                  <TableHead>水质等级</TableHead>
                  <TableHead>R²</TableHead>
                  <TableHead>MAE</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.stations.map((station) => {
                  const model = data.modelResults[station.name];
                  return (
                    <TableRow key={station.id}>
                      <TableCell className="font-medium">
                        <Link href={`/stations/${encodeURIComponent(station.name)}`} className="text-cyan-700 hover:underline">
                          {station.name}
                        </Link>
                      </TableCell>
                      <TableCell>{station.river}</TableCell>
                      <TableCell>{station.city}</TableCell>
                      <TableCell>{Number(station.latest_do).toFixed(2)}</TableCell>
                      <TableCell>
                        <Badge className={qualityBadgeColor(station.main_quality_class)}>{qualityLabel(station.main_quality_class)}</Badge>
                      </TableCell>
                      <TableCell>{model?.r2?.toFixed(4) ?? "--"}</TableCell>
                      <TableCell>{model?.mae?.toFixed(4) ?? "--"}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </section>

      <footer className="mt-6 text-xs text-muted-foreground">
        数据站点: {numberFmt.format(data.stations.reduce((sum, item) => sum + (item.data_count ?? 0), 0))} 条记录 · 预测步长: {data.metadata.forecast_horizon}
      </footer>
    </AppShell>
  );
}

function MetricCard({
  icon: Icon,
  title,
  value,
  helper,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  value: string;
  helper: string;
}) {
  return (
    <Card className="metric-tile">
      <CardContent className="flex items-center justify-between p-5">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="mt-1 text-3xl font-semibold tracking-tight">{value}</p>
          <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
        </div>
        <div className="rounded-2xl bg-cyan-100 p-3 text-cyan-800">
          <Icon className="h-6 w-6" />
        </div>
      </CardContent>
    </Card>
  );
}
