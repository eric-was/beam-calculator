import { useMemo, useState } from 'react';

const DEFAULT_SPAN = 2000; // mm
const DEFAULT_E = 10000; // MPa = N/mm^2 for MGP10
const DEFAULT_SECTION = { depth: 190, width: 45 };

const supportOptions = [
  { value: 'fixed', label: 'Fixed' },
  { value: 'pinned', label: 'Pinned' },
  { value: 'roller', label: 'Roller' },
  { value: 'free', label: 'Free' },
];

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const uniqueSorted = (values) =>
  [...new Set(values.map((value) => Number(value)))].sort((a, b) => a - b);

const beamElementStiffness = (E, I, L) => {
  const factor = (E * I) / Math.pow(L, 3);
  return [
    [12 * factor, 6 * L * factor, -12 * factor, 6 * L * factor],
    [6 * L * factor, 4 * L * L * factor, -6 * L * factor, 2 * L * L * factor],
    [-12 * factor, -6 * L * factor, 12 * factor, -6 * L * factor],
    [6 * L * factor, 2 * L * L * factor, -6 * L * factor, 4 * L * L * factor],
  ];
};

const solveLinearSystem = (matrix, vector) => {
  const size = vector.length;
  const a = matrix.map((row) => row.slice());
  const b = vector.slice();

  for (let i = 0; i < size; i += 1) {
    let maxRow = i;
    for (let k = i + 1; k < size; k += 1) {
      if (Math.abs(a[k][i]) > Math.abs(a[maxRow][i])) {
        maxRow = k;
      }
    }
    if (maxRow !== i) {
      [a[i], a[maxRow]] = [a[maxRow], a[i]];
      [b[i], b[maxRow]] = [b[maxRow], b[i]];
    }

    const pivot = a[i][i] || 1e-12;
    for (let j = i; j < size; j += 1) {
      a[i][j] /= pivot;
    }
    b[i] /= pivot;

    for (let k = 0; k < size; k += 1) {
      if (k === i) continue;
      const factor = a[k][i];
      for (let j = i; j < size; j += 1) {
        a[k][j] -= factor * a[i][j];
      }
      b[k] -= factor * b[i];
    }
  }

  return b;
};

const assembleBeamModel = ({
  span,
  E,
  I,
  supports,
  pointLoads,
  udls,
}) => {
  const nodePositions = uniqueSorted([
    0,
    span,
    ...supports.map((support) => support.position),
    ...pointLoads.map((load) => load.position),
    ...udls.flatMap((load) => [load.start, load.end]),
  ]);

  const nodeCount = nodePositions.length;
  const dofCount = nodeCount * 2;

  const K = Array.from({ length: dofCount }, () => Array(dofCount).fill(0));
  const F = Array(dofCount).fill(0);

  for (let e = 0; e < nodeCount - 1; e += 1) {
    const x1 = nodePositions[e];
    const x2 = nodePositions[e + 1];
    const L = x2 - x1;
    if (L <= 0) continue;

    const kLocal = beamElementStiffness(E, I, L);
    const dofMap = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1];

    for (let i = 0; i < 4; i += 1) {
      for (let j = 0; j < 4; j += 1) {
        K[dofMap[i]][dofMap[j]] += kLocal[i][j];
      }
    }

    const udlForElement = udls
      .filter((load) => load.start <= x1 && load.end >= x2)
      .reduce((sum, load) => sum + load.intensity, 0);

    if (udlForElement !== 0) {
      const w = udlForElement; // N/mm downward positive
      const eqForces = [
        (w * L) / 2,
        (w * L * L) / 12,
        (w * L) / 2,
        (-w * L * L) / 12,
      ];
      for (let i = 0; i < 4; i += 1) {
        F[dofMap[i]] += eqForces[i];
      }
    }
  }

  pointLoads.forEach((load) => {
    const nodeIndex = nodePositions.indexOf(load.position);
    if (nodeIndex >= 0) {
      F[nodeIndex * 2] += load.magnitude;
    }
  });

  const constrained = new Set();
  supports.forEach((support) => {
    const nodeIndex = nodePositions.indexOf(support.position);
    if (nodeIndex < 0) return;
    if (support.type === 'fixed') {
      constrained.add(nodeIndex * 2);
      constrained.add(nodeIndex * 2 + 1);
    }
    if (support.type === 'pinned' || support.type === 'roller') {
      constrained.add(nodeIndex * 2);
    }
  });

  const freeDofs = [];
  const fixedDofs = [];
  for (let i = 0; i < dofCount; i += 1) {
    if (constrained.has(i)) {
      fixedDofs.push(i);
    } else {
      freeDofs.push(i);
    }
  }

  const reducedK = freeDofs.map((row) => freeDofs.map((col) => K[row][col]));
  const reducedF = freeDofs.map((row) => F[row]);

  const freeDisplacements =
    freeDofs.length > 0 ? solveLinearSystem(reducedK, reducedF) : [];

  const displacements = Array(dofCount).fill(0);
  freeDofs.forEach((dof, index) => {
    displacements[dof] = freeDisplacements[index];
  });

  const reactions = K.map((row, i) =>
    row.reduce((sum, value, j) => sum + value * displacements[j], 0) - F[i]
  );

  return {
    nodePositions,
    displacements,
    reactions,
    K,
    F,
  };
};

const buildDiagram = ({
  span,
  nodePositions,
  displacements,
  E,
  I,
  udls,
  sample = 80,
}) => {
  const results = [];
  for (let e = 0; e < nodePositions.length - 1; e += 1) {
    const x1 = nodePositions[e];
    const x2 = nodePositions[e + 1];
    const L = x2 - x1;
    if (L <= 0) continue;

    const v1 = displacements[2 * e];
    const t1 = displacements[2 * e + 1];
    const v2 = displacements[2 * (e + 1)];
    const t2 = displacements[2 * (e + 1) + 1];

    const udlForElement = udls
      .filter((load) => load.start <= x1 && load.end >= x2)
      .reduce((sum, load) => sum + load.intensity, 0);

    const kLocal = beamElementStiffness(E, I, L);
    const localDisp = [v1, t1, v2, t2];
    const fixedEnd = [
      (udlForElement * L) / 2,
      (udlForElement * L * L) / 12,
      (udlForElement * L) / 2,
      (-udlForElement * L * L) / 12,
    ];
    const localForces = kLocal.map((row, i) =>
      row.reduce((sum, value, j) => sum + value * localDisp[j], 0) - fixedEnd[i]
    );
    const V1 = localForces[0];
    const M1 = localForces[1];

    for (let i = 0; i <= sample; i += 1) {
      const x = (i / sample) * L;
      const xi = x / L;
      const N1 = 1 - 3 * xi * xi + 2 * xi * xi * xi;
      const N2 = L * (xi - 2 * xi * xi + xi * xi * xi);
      const N3 = 3 * xi * xi - 2 * xi * xi * xi;
      const N4 = L * (-xi * xi + xi * xi * xi);
      const deflection = N1 * v1 + N2 * t1 + N3 * v2 + N4 * t2;

      const shear = V1 - udlForElement * x;
      const moment = M1 + V1 * x - (udlForElement * x * x) / 2;

      results.push({
        x: x1 + x,
        deflection,
        shear,
        moment,
      });
    }
  }

  if (results.length > 0) {
    results[results.length - 1].x = span;
  }
  return results;
};

const formatNumber = (value, digits = 2) =>
  Number.isFinite(value) ? value.toFixed(digits) : '—';

const findExtrema = (data, key) => {
  if (!data.length) {
    return {
      min: 0,
      minX: 0,
      max: 0,
      maxX: 0,
    };
  }

  return data.reduce(
    (acc, point) => {
      if (point[key] < acc.min) {
        acc.min = point[key];
        acc.minX = point.x;
      }
      if (point[key] > acc.max) {
        acc.max = point[key];
        acc.maxX = point.x;
      }
      return acc;
    },
    {
      min: data[0][key],
      minX: data[0].x,
      max: data[0][key],
      maxX: data[0].x,
    }
  );
};

const Diagram = ({ title, data, unit, color }) => {
  if (!data.length) return null;
  const width = 560;
  const height = 160;
  const padding = 20;

  const values = data.map((point) => point.value);
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1e-6);

  const points = data
    .map((point) => {
      const x = padding + (point.x / data[data.length - 1].x) * (width - 2 * padding);
      const y =
        height / 2 - (point.value / maxAbs) * (height / 2 - padding);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="diagram">
      <header>
        <h3>{title}</h3>
        <span>
          ±{formatNumber(maxAbs, 2)} {unit}
        </span>
      </header>
      <svg viewBox={`0 0 ${width} ${height}`}>
        <line
          x1={padding}
          y1={height / 2}
          x2={width - padding}
          y2={height / 2}
          stroke="#cbd5f5"
          strokeDasharray="4 4"
        />
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="2"
          points={points}
        />
      </svg>
    </div>
  );
};

const initialSupports = [
  { id: 1, position: 0, type: 'pinned' },
  { id: 2, position: 1500, type: 'roller' },
];

const initialPointLoads = [];
const initialUdls = [{ id: 1, start: 0, end: 2000, intensity: 1.5 }];

const buildSectionI = (width, depth) => (width * Math.pow(depth, 3)) / 12;

const normalizeSupports = (supports, span) =>
  supports
    .map((support) => ({
      ...support,
      position: clamp(Number(support.position), 0, span),
    }))
    .sort((a, b) => a.position - b.position);

const normalizeLoads = (loads, span, key) =>
  loads
    .map((load) => ({
      ...load,
      [key]: clamp(Number(load[key]), 0, span),
    }))
    .sort((a, b) => a[key] - b[key]);

const normalizeUdls = (udls, span) =>
  udls
    .map((load) => {
      const start = clamp(Number(load.start), 0, span);
      const end = clamp(Number(load.end), 0, span);
      return {
        ...load,
        start: Math.min(start, end),
        end: Math.max(start, end),
      };
    })
    .sort((a, b) => a.start - b.start);

const App = () => {
  const [span, setSpan] = useState(DEFAULT_SPAN);
  const [E, setE] = useState(DEFAULT_E);
  const [section, setSection] = useState(DEFAULT_SECTION);
  const [supports, setSupports] = useState(initialSupports);
  const [pointLoads, setPointLoads] = useState(initialPointLoads);
  const [udls, setUdls] = useState(initialUdls);

  const normalizedSupports = useMemo(
    () => normalizeSupports(supports, span),
    [supports, span]
  );
  const normalizedPointLoads = useMemo(
    () => normalizeLoads(pointLoads, span, 'position'),
    [pointLoads, span]
  );
  const normalizedUdls = useMemo(
    () => normalizeUdls(udls, span),
    [udls, span]
  );

  const I = useMemo(
    () => buildSectionI(section.width, section.depth),
    [section]
  );

  const beamResults = useMemo(() => {
    const model = assembleBeamModel({
      span,
      E,
      I,
      supports: normalizedSupports,
      pointLoads: normalizedPointLoads.map((load) => ({
        ...load,
        magnitude: load.magnitude * 1000,
      })),
      udls: normalizedUdls.map((load) => ({
        ...load,
        intensity: load.intensity,
      })),
    });

    const diagrams = buildDiagram({
      span,
      nodePositions: model.nodePositions,
      displacements: model.displacements,
      E,
      I,
      udls: normalizedUdls.map((load) => ({
        ...load,
        intensity: load.intensity,
      })),
    });

    return { ...model, diagrams };
  }, [span, E, I, normalizedSupports, normalizedPointLoads, normalizedUdls]);

  const diagramData = useMemo(() => {
    return beamResults.diagrams.map((point) => ({
      x: point.x,
      deflection: point.deflection,
      moment: point.moment / 1e6,
      shear: point.shear / 1000,
    }));
  }, [beamResults.diagrams]);

  const supportReactions = useMemo(() => {
    return normalizedSupports.map((support) => {
      const nodeIndex = beamResults.nodePositions.indexOf(support.position);
      const reaction =
        nodeIndex >= 0 ? beamResults.reactions[nodeIndex * 2] / 1000 : 0;
      const moment =
        nodeIndex >= 0 && support.type === 'fixed'
          ? beamResults.reactions[nodeIndex * 2 + 1] / 1e6
          : 0;
      return { ...support, reaction, moment };
    });
  }, [normalizedSupports, beamResults]);

  const shearExtrema = useMemo(
    () => findExtrema(diagramData, 'shear'),
    [diagramData]
  );
  const momentExtrema = useMemo(
    () => findExtrema(diagramData, 'moment'),
    [diagramData]
  );
  const deflectionExtrema = useMemo(
    () => findExtrema(diagramData, 'deflection'),
    [diagramData]
  );

  const isCantileverStart =
    normalizedSupports.length === 0 || normalizedSupports[0].position > 0;
  const isCantileverEnd =
    normalizedSupports.length === 0 ||
    normalizedSupports[normalizedSupports.length - 1].position < span;

  const handleSupportChange = (id, key, value) => {
    setSupports((items) =>
      items.map((item) =>
        item.id === id ? { ...item, [key]: value } : item
      )
    );
  };

  const handleLoadChange = (setState) => (id, key, value) => {
    setState((items) =>
      items.map((item) =>
        item.id === id ? { ...item, [key]: value } : item
      )
    );
  };

  const addSupport = () =>
    setSupports((items) => [
      ...items,
      { id: Date.now(), position: span, type: 'roller' },
    ]);

  const addPointLoad = () =>
    setPointLoads((items) => [
      ...items,
      { id: Date.now(), position: span / 2, magnitude: 5 },
    ]);

  const addUdl = () =>
    setUdls((items) => [
      ...items,
      { id: Date.now(), start: 0, end: span, intensity: 1 },
    ]);

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="tag">Prismatic Euler–Bernoulli Beam</p>
          <h1>Beam reaction, shear, moment & deflection explorer</h1>
          <p>
            Define support locations (fixed, pinned, roller, or free), apply point
            loads and UDLs, and instantly preview diagrams. The model uses a
            standard 2‑DOF Euler–Bernoulli beam finite element formulation with
            consistent load vectors.
          </p>
        </div>
        <div className="spec-card">
          <h2>Member preset</h2>
          <div className="spec-row">
            <span>Section</span>
            <strong>190 × 45 MGP10</strong>
          </div>
          <div className="spec-row">
            <span>Span</span>
            <strong>{span} mm</strong>
          </div>
          <div className="spec-row">
            <span>E</span>
            <strong>{E} MPa</strong>
          </div>
          <div className="spec-row">
            <span>I</span>
            <strong>{formatNumber(I / 1e6, 2)} ×10⁶ mm⁴</strong>
          </div>
          <div className="hint">
            Loads are downward positive. Reactions may appear negative (upwards).
          </div>
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>Geometry & material</h2>
        </div>
        <div className="grid">
          <label>
            Span (mm)
            <input
              type="number"
              value={span}
              min="1"
              onChange={(event) => setSpan(Number(event.target.value))}
            />
          </label>
          <label>
            E (MPa)
            <input
              type="number"
              value={E}
              onChange={(event) => setE(Number(event.target.value))}
            />
          </label>
          <label>
            Depth (mm)
            <input
              type="number"
              value={section.depth}
              onChange={(event) =>
                setSection((value) => ({
                  ...value,
                  depth: Number(event.target.value),
                }))
              }
            />
          </label>
          <label>
            Width (mm)
            <input
              type="number"
              value={section.width}
              onChange={(event) =>
                setSection((value) => ({
                  ...value,
                  width: Number(event.target.value),
                }))
              }
            />
          </label>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Supports</h2>
          <button type="button" className="ghost" onClick={addSupport}>
            + Add support
          </button>
        </div>
        <div className="list">
          {normalizedSupports.map((support) => (
            <div className="list-row" key={support.id}>
              <input
                type="number"
                value={support.position}
                onChange={(event) =>
                  handleSupportChange(
                    support.id,
                    'position',
                    Number(event.target.value)
                  )
                }
              />
              <select
                value={support.type}
                onChange={(event) =>
                  handleSupportChange(support.id, 'type', event.target.value)
                }
              >
                {supportOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="danger"
                onClick={() =>
                  setSupports((items) =>
                    items.filter((item) => item.id !== support.id)
                  )
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
        <div className="status">
          <span>
            {isCantileverStart
              ? 'Free end at x = 0 mm (cantilever detected)'
              : 'Support at start'}
          </span>
          <span>
            {isCantileverEnd
              ? `Free end at x = ${span} mm (cantilever detected)`
              : 'Support at end'}
          </span>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Point loads (kN)</h2>
          <button type="button" className="ghost" onClick={addPointLoad}>
            + Add point load
          </button>
        </div>
        <div className="list">
          {normalizedPointLoads.map((load) => (
            <div className="list-row" key={load.id}>
              <input
                type="number"
                value={load.position}
                onChange={(event) =>
                  handleLoadChange(setPointLoads)(
                    load.id,
                    'position',
                    Number(event.target.value)
                  )
                }
              />
              <input
                type="number"
                value={load.magnitude}
                onChange={(event) =>
                  handleLoadChange(setPointLoads)(
                    load.id,
                    'magnitude',
                    Number(event.target.value)
                  )
                }
              />
              <button
                type="button"
                className="danger"
                onClick={() =>
                  setPointLoads((items) =>
                    items.filter((item) => item.id !== load.id)
                  )
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>UDLs (kN/m)</h2>
          <button type="button" className="ghost" onClick={addUdl}>
            + Add UDL
          </button>
        </div>
        <div className="list">
          {normalizedUdls.map((load) => (
            <div className="list-row" key={load.id}>
              <input
                type="number"
                value={load.start}
                onChange={(event) =>
                  handleLoadChange(setUdls)(
                    load.id,
                    'start',
                    Number(event.target.value)
                  )
                }
              />
              <input
                type="number"
                value={load.end}
                onChange={(event) =>
                  handleLoadChange(setUdls)(
                    load.id,
                    'end',
                    Number(event.target.value)
                  )
                }
              />
              <input
                type="number"
                value={load.intensity}
                onChange={(event) =>
                  handleLoadChange(setUdls)(
                    load.id,
                    'intensity',
                    Number(event.target.value)
                  )
                }
              />
              <button
                type="button"
                className="danger"
                onClick={() =>
                  setUdls((items) => items.filter((item) => item.id !== load.id))
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Reactions</h2>
        </div>
        <div className="grid reactions">
          {supportReactions.map((support) => (
            <div key={support.id} className="reaction-card">
              <h3>
                {support.type} @ {support.position} mm
              </h3>
              <p>
                V = {formatNumber(support.reaction, 2)} kN
                {support.type === 'fixed' && (
                  <>
                    <br />M = {formatNumber(support.moment, 2)} kN·m
                  </>
                )}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Diagrams</h2>
        </div>
        <div className="diagram-grid">
          <Diagram
            title="Shear"
            unit="kN"
            color="#4f46e5"
            data={diagramData.map((point) => ({
              x: point.x,
              value: point.shear,
            }))}
          />
          <Diagram
            title="Moment"
            unit="kN·m"
            color="#db2777"
            data={diagramData.map((point) => ({
              x: point.x,
              value: point.moment,
            }))}
          />
          <Diagram
            title="Deflection"
            unit="mm"
            color="#059669"
            data={diagramData.map((point) => ({
              x: point.x,
              value: point.deflection,
            }))}
          />
        </div>
        <div className="extrema-grid">
          <div>
            <h3>Shear extrema</h3>
            <p>
              Max: {formatNumber(shearExtrema.max, 2)} kN @{' '}
              {formatNumber(shearExtrema.maxX, 0)} mm
              <br />
              Min: {formatNumber(shearExtrema.min, 2)} kN @{' '}
              {formatNumber(shearExtrema.minX, 0)} mm
            </p>
          </div>
          <div>
            <h3>Moment extrema</h3>
            <p>
              Max: {formatNumber(momentExtrema.max, 2)} kN·m @{' '}
              {formatNumber(momentExtrema.maxX, 0)} mm
              <br />
              Min: {formatNumber(momentExtrema.min, 2)} kN·m @{' '}
              {formatNumber(momentExtrema.minX, 0)} mm
            </p>
          </div>
          <div>
            <h3>Deflection extrema</h3>
            <p>
              Max: {formatNumber(deflectionExtrema.max, 3)} mm @{' '}
              {formatNumber(deflectionExtrema.maxX, 0)} mm
              <br />
              Min: {formatNumber(deflectionExtrema.min, 3)} mm @{' '}
              {formatNumber(deflectionExtrema.minX, 0)} mm
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default App;
