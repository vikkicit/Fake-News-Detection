"use client";

import React, { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";

interface HeadlineItem {
  text: string;
  verdict: "ALERT" | "VERIFIED" | "SCANNING";
  radius: number;
  speed: number;
  initialAngle: number;
  yOffset: number;
}

const HEADLINES: HeadlineItem[] = [
  {
    text: "BREAKING: Viral Election Claims Debunked by Forensic AI",
    verdict: "ALERT",
    radius: 3.8,
    speed: 0.12,
    initialAngle: 0,
    yOffset: 1.2,
  },
  {
    text: "Municipal Infrastructure Report Confirmed Official Source",
    verdict: "VERIFIED",
    radius: 4.2,
    speed: -0.09,
    initialAngle: 1.8,
    yOffset: -0.8,
  },
  {
    text: "Deepfake Speech Vector Detected in Global Stream #409",
    verdict: "ALERT",
    radius: 4.0,
    speed: 0.14,
    initialAngle: 3.5,
    yOffset: 0.4,
  },
  {
    text: "Institutional Financial Disclosures Cross-Matched",
    verdict: "VERIFIED",
    radius: 3.6,
    speed: -0.11,
    initialAngle: 5.1,
    yOffset: -1.3,
  },
  {
    text: "Synthetic Sentiment Propagation Pattern Flagged",
    verdict: "SCANNING",
    radius: 4.5,
    speed: 0.08,
    initialAngle: 2.7,
    yOffset: 1.7,
  },
];

function OrbitingHeadline({ item }: { item: HeadlineItem }) {
  const groupRef = useRef<THREE.Group>(null!);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    const angle = item.initialAngle + t * item.speed;
    const x = Math.cos(angle) * item.radius;
    const z = Math.sin(angle) * item.radius;
    const y = item.yOffset + Math.sin(t * 0.8 + item.initialAngle) * 0.2;

    if (groupRef.current) {
      groupRef.current.position.set(x, y, z);
    }
  });

  const badgeColor =
    item.verdict === "ALERT"
      ? "bg-red-500/20 text-red-400 border-red-500/50"
      : item.verdict === "VERIFIED"
      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/50"
      : "bg-cyan-500/20 text-cyan-400 border-cyan-500/50";

  return (
    <group ref={groupRef}>
      <Html distanceFactor={11} center className="pointer-events-none select-none">
        <div className="backdrop-blur-md bg-black/60 border border-white/10 rounded-lg p-2.5 shadow-2xl flex items-center space-x-2.5 whitespace-nowrap min-w-[260px] text-xs">
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono tracking-wider border ${badgeColor}`}
          >
            {item.verdict}
          </span>
          <span className="text-gray-200 font-medium tracking-wide">
            {item.text}
          </span>
        </div>
      </Html>
    </group>
  );
}

export function FloatingHeadlines() {
  return (
    <group>
      {HEADLINES.map((item, index) => (
        <OrbitingHeadline key={index} item={item} />
      ))}
    </group>
  );
}
