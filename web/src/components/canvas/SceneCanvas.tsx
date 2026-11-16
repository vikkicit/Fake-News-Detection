"use client";

import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { HolographicGlobe } from "./HolographicGlobe";
import { DataStreams, AmbientParticles } from "./DataStreams";

function CameraRig({ scrollProgress }: { scrollProgress: number }) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null!);

  useFrame(() => {
    if (!cameraRef.current) return;

    // Camera movement based on scroll progress (0 to 1)
    // Section 1: Intro/Globe View (x=0, y=0.5, z=7.5)
    // Section 2: News Investigation (orbit around left, zoom slightly in)
    // Section 3: AI Processing (pan downward angled into core)
    // Section 4: Prediction Results (elevated cinematic perspective)

    const p = scrollProgress;

    const targetX = Math.sin(p * Math.PI * 1.5) * 3.5;
    const targetY = 0.5 - p * 2.2 + Math.cos(p * Math.PI) * 0.8;
    const targetZ = 7.5 - Math.sin(p * Math.PI) * 2.0;

    cameraRef.current.position.x = THREE.MathUtils.lerp(
      cameraRef.current.position.x,
      targetX,
      0.05
    );
    cameraRef.current.position.y = THREE.MathUtils.lerp(
      cameraRef.current.position.y,
      targetY,
      0.05
    );
    cameraRef.current.position.z = THREE.MathUtils.lerp(
      cameraRef.current.position.z,
      targetZ,
      0.05
    );

    cameraRef.current.lookAt(0, -p * 0.5, 0);
  });

  return (
    <PerspectiveCamera
      ref={cameraRef}
      makeDefault
      position={[0, 0.5, 7.5]}
      fov={48}
    />
  );
}

export function SceneCanvas({ scrollProgress }: { scrollProgress: number }) {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 w-full h-full bg-[#020208]">
      <Canvas
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#020208"]} />
        <fog attach="fog" args={["#020208", 8, 22]} />

        <CameraRig scrollProgress={scrollProgress} />

        {/* Ambient & Volumetric Light Simulation */}
        <ambientLight intensity={0.25} />
        <directionalLight position={[10, 10, 5]} intensity={1.2} color="#00f0ff" />
        <pointLight position={[-10, -5, -5]} intensity={1.8} color="#a855f7" />
        <pointLight position={[0, 0, 0]} intensity={2.5} color="#00f0ff" distance={6} />

        {/* 3D Scene Elements */}
        <HolographicGlobe />
        <DataStreams />
        <AmbientParticles count={1000} />
      </Canvas>
    </div>
  );
}
