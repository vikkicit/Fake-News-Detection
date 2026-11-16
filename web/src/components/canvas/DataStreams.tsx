"use client";

import React, { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export function DataStreams() {
  const linesGroupRef = useRef<THREE.Group>(null!);

  // Generate curved 3D stream lines orbiting around globe
  const curves = useMemo(() => {
    const list = [];
    const colors = ["#00f0ff", "#3b82f6", "#8b5cf6", "#ec4899"];

    for (let i = 0; i < 18; i++) {
      const radius = 2.4 + Math.random() * 1.8;
      const angleOffset = Math.random() * Math.PI * 2;
      const height = (Math.random() - 0.5) * 3.5;

      const points = [];
      const segments = 24;
      for (let s = 0; s <= segments; s++) {
        const theta = angleOffset + (s / segments) * Math.PI * 0.9;
        const x = Math.cos(theta) * radius;
        const z = Math.sin(theta) * radius;
        const y = height + Math.sin(s * 0.4) * 0.6;
        points.push(new THREE.Vector3(x, y, z));
      }

      const curve = new THREE.CatmullRomCurve3(points);
      const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(50));
      const color = colors[i % colors.length];
      list.push({ geom, color, speed: 0.15 + Math.random() * 0.2 });
    }
    return list;
  }, []);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (linesGroupRef.current) {
      linesGroupRef.current.rotation.y = t * 0.08;
      linesGroupRef.current.rotation.x = Math.sin(t * 0.05) * 0.05;
    }
  });

  return (
    <group ref={linesGroupRef}>
      {curves.map((item, index) => (
        <primitive
          key={index}
          object={
            new THREE.Line(
              item.geom,
              new THREE.LineBasicMaterial({
                color: item.color,
                transparent: true,
                opacity: 0.4,
                blending: THREE.AdditiveBlending,
              })
            )
          }
        />
      ))}
    </group>
  );
}

export function AmbientParticles({ count = 1200 }: { count?: number }) {
  const pointsRef = useRef<THREE.Points>(null!);

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const c1 = new THREE.Color("#00f0ff");
    const c2 = new THREE.Color("#9333ea");

    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 35;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 35;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 25;

      const mixed = c1.clone().lerp(c2, Math.random());
      col[i * 3] = mixed.r;
      col[i * 3 + 1] = mixed.g;
      col[i * 3 + 2] = mixed.b;
    }
    return [pos, col];
  }, [count]);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = t * 0.02;
      pointsRef.current.rotation.x = t * 0.01;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.07}
        vertexColors
        transparent
        opacity={0.7}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
