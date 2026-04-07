"use client";

import Link from "next/link";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AppShell } from "@/components/dashboard/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ModelResultNode, StationForecast, StationNode } from "@/lib/data";
import { qualityBadgeColor, qualityLabel } from "@/lib/quality";

type StationDetailClientProps = {
  station: StationNode;
  forecast?: StationForecast;
  model?: ModelResultNode;
};

export function StationDetailClient({ station, forecast, model }: StationDetailClientProps) {
  const trendData = [
    ...(forecast?.recent_history.map((item) => ({
      time: item.time.slice(5),
      actual: item.do,
      forecast: undefined,
    })) ?? []),
    {
      time: forecast?.current_time?.slice(5) ?? "--",
      actual: forecast?.current_do ?? undefined,
      forecast: forecast?.current_do ?? undefined,
    },
    ...((forecast?.forecasts ?? []).map((item) => ({
      time: item.time,
      actual: undefined,
      forecast: item.do,
    })) as Array<{ time: string; actual?: number; forecast?: number }>),
  ];

  const featureData =
    model?.top_features?.slice(0, 8).map((item) => ({
      name: item.name,
      value: Number(item.importance.toFixed(4)),
    })) ?? [];

  return (
    <AppShell
      title={`${station.name} 站点详情`}
      subtitle={`${station.river} · ${station.city}，展示当前状态、趋势预测与模型解释。`}
    >
      <section className="mt-6 grid gap-4 md:grid-cols-3">
        <Card className="metric-tile">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">当前 DO</p>
            <p className="mt-1 text-3xl font-semibold">{forecast?.current_do?.toFixed(2) ?? station.latest_do.toFixed(2)} mg/L</p>
          </CardContent>
        </Card>
        <Card className="metric-tile">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">水质等级</p>
            <div className="mt-2">
              <Badge className={qualityBadgeColor(forecast?.current_quality ?? station.main_quality_class)}>
                {qualityLabel(forecast?.current_quality ?? station.main_quality_class)}
              </Badge>
            </div>
          </CardContent>
        </Card>
        <Card className="metric-tile">
          <CardContent className="p-5">
            <p className="text-sm text-muted-foreground">模型 R² / MAE</p>
            <p className="mt-1 text-3xl font-semibold">
              {model?.r2?.toFixed(3) ?? "--"} / {model?.mae?.toFixed(3) ?? "--"}
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="mt-6 grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>DO 趋势</CardTitle>
            <CardDescription>近 7 天实测 + 未来 24h 预测</CardDescription>
          </CardHeader>
          <CardContent className="h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#bae6fd" />
                <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 11 }} unit=" mg/L" />
                <Tooltip />
                <Area type="monotone" dataKey="actual" stroke="#0284c7" fillOpacity={0.18} fill="#0ea5e9" name="实测" />
                <Area type="monotone" dataKey="forecast" stroke="#0f766e" fillOpacity={0.2} fill="#2dd4bf" name="预测" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>驱动因子</CardTitle>
            <CardDescription>Top 8 特征重要性</CardDescription>
          </CardHeader>
          <CardContent className="h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureData} layout="vertical" margin={{ left: 20, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbeafe" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#0f766e" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      <section className="mt-6">
        <Card className="border-white/60 bg-white/95">
          <CardHeader>
            <CardTitle>基础信息</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm md:grid-cols-2">
            <div>省份: {station.province}</div>
            <div>城市: {station.city}</div>
            <div>河流: {station.river}</div>
            <div>经纬度: {station.lon.toFixed(4)}, {station.lat.toFixed(4)}</div>
            <div>数据条数: {station.data_count}</div>
            <div>时间范围: {station.data_period}</div>
          </CardContent>
        </Card>
      </section>

      <div className="mt-6 flex gap-3">
        <Button variant="outline" asChild>
          <Link href="/">返回总览</Link>
        </Button>
        <Button className="bg-cyan-700 hover:bg-cyan-600" asChild>
          <Link href="/insights">查看解释分析</Link>
        </Button>
      </div>
    </AppShell>
  );
}
