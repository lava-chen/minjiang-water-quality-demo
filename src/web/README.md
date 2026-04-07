# Minjiang Web Dashboard

## 1. Install

```bash
cd src/web
npm install
```

## 2. Run

```bash
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## 3. Stack

- Next.js (App Router, TypeScript)
- Tailwind CSS
- shadcn/ui style components
- Recharts for data visualization

## 4. Data Source

- `src/knowledge_graph/kg_data.json`
- `models/forecasts.json`

## 5. Pages

- `/` 总览看板（KPI、空间分布、趋势、站点总览）
- `/stations/[name]` 站点详情页
- `/insights` 全局解释分析页
