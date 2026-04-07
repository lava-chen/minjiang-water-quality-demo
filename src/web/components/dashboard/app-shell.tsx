"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type ShellProps = {
  title: string;
  subtitle: string;
  children: ReactNode;
};

const navItems = [
  { href: "/", label: "总览" },
  { href: "/insights", label: "解释分析" },
];

export function AppShell({ title, subtitle, children }: ShellProps) {
  const pathname = usePathname();

  return (
    <div className="container py-6 md:py-10">
      <header className="animate-fade-up">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <p className="inline-flex rounded-full bg-cyan-900/90 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-50">
            Minjiang Intelligence Dashboard
          </p>
          <nav className="flex gap-2 rounded-xl border border-white/70 bg-white/70 p-1 backdrop-blur">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-sm text-slate-700 transition-colors",
                  pathname === item.href ? "bg-cyan-700 text-white" : "hover:bg-cyan-50",
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight md:text-4xl">{title}</h1>
        <p className="mt-3 max-w-3xl text-sm text-muted-foreground md:text-base">{subtitle}</p>
      </header>
      {children}
    </div>
  );
}
