import { useEffect, useRef } from "react";
import * as THREE from "three";
import * as d3 from "d3";
import type { Helix3DDoc } from "../../lib/types";

interface Props {
  data: Helix3DDoc;
  height?: number;
  showLabels?: boolean;
}

/**
 * Three.js-rendered 3D helix. Drag to rotate (orbit on a stable Y axis,
 * tilt up/down). No external OrbitControls dependency -- written inline so
 * the bundle stays under 200 KB.
 *
 * Mapping: data.coords[i] -> (x, y, z) point in the QR-orthonormalised
 *   (u_cos, u_sin, u_lin) frame of the residual stream. Linear axis is z.
 */
export function HelixViewer3D({ data, height = 480, showLabels = true }: Props) {
  const mount = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mount.current) return;
    const el = mount.current;
    const w = el.clientWidth;
    const h = height;

    // Scene + camera + renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#ffffff");

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    camera.position.set(3.2, 2.4, 3.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    el.appendChild(renderer.domElement);

    // Normalise coords into a unit cube centred at the origin.
    const coords = data.coords;
    const bounds = (axis: 0 | 1 | 2) => {
      const xs = coords.map((p) => p[axis]);
      const lo = Math.min(...xs), hi = Math.max(...xs);
      return { lo, hi, mid: (lo + hi) / 2, range: hi - lo || 1 };
    };
    const bx = bounds(0), by = bounds(1), bz = bounds(2);
    const scale = 2 / Math.max(bx.range, by.range, bz.range);
    const toScene = (p: [number, number, number]): [number, number, number] => [
      (p[0] - bx.mid) * scale,
      (p[2] - bz.mid) * scale,  // linear axis is "up" in scene (Y)
      (p[1] - by.mid) * scale,
    ];
    const scenePts = coords.map(toScene);

    // Axes (subtle grey crosshair)
    const axesGroup = new THREE.Group();
    const ax = (dir: THREE.Vector3, len = 1.4) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), dir.clone().multiplyScalar(len)]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xbfbfbf }));
    };
    axesGroup.add(ax(new THREE.Vector3(1,0,0)));
    axesGroup.add(ax(new THREE.Vector3(0,1,0)));
    axesGroup.add(ax(new THREE.Vector3(0,0,1)));
    scene.add(axesGroup);

    // Connecting trajectory line (the helix shape)
    const lineGeo = new THREE.BufferGeometry().setFromPoints(
      scenePts.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
    );
    const lineMat = new THREE.LineBasicMaterial({ color: 0x9ca3af, transparent: true, opacity: 0.6 });
    scene.add(new THREE.Line(lineGeo, lineMat));

    // Points coloured by integer value (viridis)
    const nMax = Math.max(...data.numbers, 1);
    const pointsGroup = new THREE.Group();
    scenePts.forEach(([x, y, z], i) => {
      const color = new THREE.Color(d3.interpolateViridis(data.numbers[i] / nMax));
      const mat = new THREE.MeshBasicMaterial({ color });
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 16), mat);
      sphere.position.set(x, y, z);
      pointsGroup.add(sphere);
    });
    scene.add(pointsGroup);

    // Optional labels using CSS-side overlay; cheaper than text sprites for 100 points.
    const labelLayer = document.createElement("div");
    labelLayer.style.cssText = `position:absolute;inset:0;pointer-events:none;font-family:ui-monospace,monospace;font-size:9px;color:#1f2937;`;
    el.style.position = "relative";
    el.appendChild(labelLayer);
    const labelNodes: HTMLSpanElement[] = [];
    if (showLabels) {
      scenePts.forEach((_, i) => {
        const s = document.createElement("span");
        s.style.cssText = "position:absolute;transform:translate(-50%,-50%);background:#ffffffc0;padding:0 2px;border-radius:2px;";
        s.textContent = data.labels[i];
        labelLayer.appendChild(s);
        labelNodes.push(s);
      });
    }

    function project(p: [number, number, number]): [number, number] {
      const v = new THREE.Vector3(p[0], p[1], p[2]);
      v.project(camera);
      return [(v.x + 1) / 2 * w, (1 - v.y) / 2 * h];
    }
    function updateLabels() {
      if (!showLabels) return;
      scenePts.forEach((p, i) => {
        const [px, py] = project(p);
        labelNodes[i].style.left = `${px}px`;
        labelNodes[i].style.top  = `${py - 10}px`;
      });
    }

    // Simple orbit interaction (no external dep)
    let dragging = false;
    let lastX = 0, lastY = 0;
    let theta = Math.PI / 4, phi = Math.PI / 5, radius = camera.position.length();
    function applyCamera() {
      const x = radius * Math.cos(phi) * Math.sin(theta);
      const y = radius * Math.sin(phi);
      const z = radius * Math.cos(phi) * Math.cos(theta);
      camera.position.set(x, y, z);
      camera.lookAt(0, 0, 0);
    }
    applyCamera();
    const onDown = (e: PointerEvent) => { dragging = true; lastX = e.clientX; lastY = e.clientY; };
    const onUp = () => { dragging = false; };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      theta -= dx * 0.005;
      phi = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, phi - dy * 0.005));
      applyCamera();
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      radius = Math.max(1.6, Math.min(8, radius * (1 + e.deltaY * 0.001)));
      applyCamera();
    };
    renderer.domElement.style.cursor = "grab";
    renderer.domElement.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    // Animation loop: gentle auto-rotate until the user drags.
    let raf = 0;
    let touched = false;
    renderer.domElement.addEventListener("pointerdown", () => { touched = true; }, { once: true });
    const tick = () => {
      if (!touched) {
        theta += 0.002;
        applyCamera();
      }
      renderer.render(scene, camera);
      updateLabels();
      raf = requestAnimationFrame(tick);
    };
    tick();

    // Resize
    const onResize = () => {
      const w2 = el.clientWidth;
      renderer.setSize(w2, h);
      camera.aspect = w2 / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("wheel", onWheel);
      el.removeChild(renderer.domElement);
      el.removeChild(labelLayer);
      renderer.dispose();
    };
  }, [data, height, showLabels]);

  return <div ref={mount} style={{ height }} className="w-full" />;
}
