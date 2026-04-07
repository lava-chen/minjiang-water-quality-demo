import { notFound } from "next/navigation";

import { StationDetailClient } from "@/components/dashboard/station-detail-client";
import { getStationDetail } from "@/lib/data";

type PageProps = {
  params: Promise<{ name: string }>;
};

export default async function StationPage({ params }: PageProps) {
  const { name } = await params;
  const detail = getStationDetail(name);

  if (!detail) {
    notFound();
  }

  return <StationDetailClient station={detail.station} forecast={detail.forecast} model={detail.model} />;
}
