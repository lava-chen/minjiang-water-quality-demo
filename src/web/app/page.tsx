import { DashboardClient } from "@/components/dashboard/dashboard-client";
import { getDashboardData } from "@/lib/data";

export default function HomePage() {
  const data = getDashboardData();
  return <DashboardClient data={data} />;
}
