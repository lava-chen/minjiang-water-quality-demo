"use client";

import Link from "next/link";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AppShell } from "@/components/dashboard/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardData } from "@/lib/data";

type InsightsClientProps = {
  data: DashboardData;
};

export function InsightsClient({ data }: InsightsClientProps) {
  const ranking = Object.values(data.modelResults)
    .sort((a, b) => b.r2 - a.r2)
    .map((item) => ({
      station: item.station,
      r2: Number(item.r2.toFixed(4)),
      mae: Number(item.mae.toFixed(4)),
    }));

  const featureFrequency = new Map<string, number>();
  Object.values(data.modelResults).forEach((item) => {
    item.top_features?.slice(0, 5).forEach((feature) => {
      featureFrequency.set(feature.name, (featureFrequency.get(feature.name) ?? 0) + 1);
    });
  });

  const dominantFeatures = [...featureFrequency.entries()]
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  return (
    <AppShell
      title="模型解释分析"
      subtitle="从全局模型质量、关键因子频率与站点差异角度，快速识别高风险站点与共性驱动因素。"
    >
      <section className="mt-6 grid gap-4 xl:grid-cols-2">
        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>站点模型质量排名（R²）</CardTitle>
            <CardDescription>按拟合优度从高到低排序</CardDescription>
          </CardHeader>
          <CardContent className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ranking} layout="vertical" margin={{ left: 20, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbeafe" />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="station" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="r2" fill="#0e7490" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-white/60 bg-white/90">
          <CardHeader>
            <CardTitle>高频驱动因子</CardTitle>
            <CardDescription>统计各站点 Top5 中出现频率最高的特征</CardDescription>
          </CardHeader>
          <CardContent className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dominantFeatures} layout="vertical" margin={{ left: 20, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbeafe" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
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
            <CardTitle>站点详情快捷入口</CardTitle>
            <CardDescription>点击进入对应站点详情页</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {data.stations
              .filter((item) => item.is_key)
              .map((item) => (
                <Button key={item.id} variant="outline" className="border-cyan-200" asChild>
                  <Link href={`/stations/${encodeURIComponent(item.name)}`}>{item.name}</Link>
                </Button>
              ))}
          </CardContent>
        </Card>
      </section>
    </AppShell>
  );
}
