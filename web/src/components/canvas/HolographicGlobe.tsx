"use client";

import React, { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export function HolographicGlobe() {
  const outerSphereRef = useRef<THREE.Mesh>(null!);
  const innerSphereRef = useRef<THREE.Mesh>(null!);
  const wireframeRef = useRef<THREE.LineSegments>(null!);
  const ring1Ref = useRef<THREE.Mesh>(null!);
  const ring2Ref = useRef<THREE.Mesh>(null!);
  const particlesRef = useRef<THREE.Points>(null!);

  // Generate globe surface particle clouds
  const [particlePositions, particleColors] = useMemo(() => {
    const count = 1800;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    const color1 = new THREE.Color("#00f0ff");
    const color2 = new THREE.Color("#7928ca");

    for (let i = 0; i < count; i++) {
      const phi = Math.acos(-1 + (2 * i) / count);
      const theta = Math.sqrt(count * Math.PI) * phi;
      const radius = 2.15 + (Math.random() - 0.5) * 0.15;

      positions[i * 3] = radius * Math.cos(theta) * Math.sin(phi);
      positions[i * 3 + 1] = radius * Math.sin(theta) * Math.sin(phi);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      const mixed = color1.clone().lerp(color2, Math.random());
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;
    }
    return [positions, colors];
  }, []);

  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime();
    if (outerSphereRef.current) {
      outerSphereRef.current.rotation.y = t * 0.12;
      outerSphereRef.current.rotation.x = Math.sin(t * 0.05) * 0.1;
    }
    if (innerSphereRef.current) {
      innerSphereRef.current.rotation.y = -t * 0.18;
    }
    if (wireframeRef.current) {
      wireframeRef.current.rotation.y = t * 0.08;
    }
    if (ring1Ref.current) {
      ring1Ref.current.rotation.z = t * 0.25;
      ring1Ref.current.rotation.x = Math.PI / 2.6 + Math.sin(t * 0.2) * 0.1;
    }
    if (ring2Ref.current) {
      ring2Ref.current.rotation.z = -t * 0.2;
      ring2Ref.current.rotation.y = Math.PI / 4 + Math.cos(t * 0.15) * 0.15;
    }
    if (particlesRef.current) {
      particlesRef.current.rotation.y = t * 0.05;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Inner Glowing Core */}
      <mesh ref={innerSphereRef}>
        <sphereGeometry args={[1.5, 32, 32]} />
        <meshBasicMaterial
          color="#7928ca"
          transparent
          opacity={0.35}
          wireframe
        />
      </mesh>

      {/* Main Holographic Hologram Shell */}
      <mesh ref={outerSphereRef}>
        <sphereGeometry args={[2.0, 48, 48]} />
        <meshStandardMaterial
          color="#001830"
          emissive="#00f0ff"
          emissiveIntensity={0.4}
          roughness={0.2}
          metalness={0.9}
          transparent
          opacity={0.65}
        />
      </mesh>

      {/* Hexagonal / Segment Wireframe Grid */}
      <lineSegments ref={wireframeRef}>
        <wireframeGeometry args={[new THREE.IcosahedronGeometry(2.05, 3)]} />
        <lineBasicMaterial color="#00f0ff" transparent opacity={0.3} />
      </lineSegments>

      {/* Globe Node Particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particlePositions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[particleColors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.045}
          vertexColors
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Orbital Data Ring 1 (Cyan) */}
      <mesh ref={ring1Ref}>
        <ringGeometry args={[2.7, 2.76, 64]} />
        <meshBasicMaterial
          color="#00f0ff"
          side={THREE.DoubleSide}
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* Orbital Data Ring 2 (Purple) */}
      <mesh ref={ring2Ref}>
        <ringGeometry args={[3.1, 3.14, 64]} />
        <meshBasicMaterial
          color="#a855f7"
          side={THREE.DoubleSide}
          transparent
          opacity={0.45}
        />
      </mesh>
    </group>
  );
}
