"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { NavigationHeader, OverlaySections } from "@/components/dom/OverlaySections";

// Dynamically import Three.js scene canvas with SSR disabled
const SceneCanvas = dynamic(
  () => import("@/components/canvas/SceneCanvas").then((mod) => mod.SceneCanvas),
  { ssr: false }
);

export default function Home() {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = totalHeight > 0 ? scrollY / totalHeight : 0;
      setScrollProgress(Math.min(Math.max(progress, 0), 1));
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <main className="relative min-h-screen bg-[#020208] text-white overflow-x-hidden">
      {/* 3D Background Canvas */}
      <SceneCanvas scrollProgress={scrollProgress} />

      {/* Fixed Navigation Bar */}
      <NavigationHeader />

      {/* Scrollable DOM Content & Story Sections */}
      <OverlaySections />
    </main>
  );
}
