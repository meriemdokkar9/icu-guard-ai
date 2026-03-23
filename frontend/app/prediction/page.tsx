"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

type LatestVitals = {
  heart_rate?: number | null;
  spo2?: number | null;
  respiratory_rate?: number | null;
  systolic_bp?: number | null;
  lactate?: number | null;
  oxygen_device?: string | null;
  gender?: string | null;
  admission_type?: string | null;
  latest_hour_from_panel?: number | null;
};

type PatientCard = {
  patient_id: number;
  risk_score: number;
  risk_level: string;
  lead_time_hours: number;
  threshold_used?: number;
  latest_hour_from_admission?: number;
  true_label_if_available?: number;
  alert_banner?: string;
  recommendation?: string;
  latest_vitals?: LatestVitals;
  message?: string;
};

type DashboardResponse = {
  last_updated_utc: string;
  total_patients: number;
  high_risk_count: number;
  moderate_risk_count: number;
  low_risk_count: number;
  active_alerts: number;
  patients: PatientCard[];
};

type TimeSeriesResponse = {
  patient_id: number;
  hours: Array<number | null>;
  heart_rate: Array<number | null>;
  respiratory_rate: Array<number | null>;
  systolic_bp: Array<number | null>;
  spo2: Array<number | null>;
  lactate: Array<number | null>;
};

function SparklineCard({
  title,
  values,
  colorClass,
}: {
  title: string;
  values: Array<number | null>;
  colorClass: string;
}) {
  const numericValues = values.filter((v): v is number => typeof v === "number");

  if (numericValues.length < 2) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="text-sm text-slate-500">{title}</p>
        <div className="mt-4 text-sm text-slate-400">No sufficient data</div>
      </div>
    );
  }

  const width = 260;
  const height = 90;
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const range = max - min || 1;

  const points = values
    .map((v, i) => {
      if (typeof v !== "number") return null;
      const x = (i / Math.max(values.length - 1, 1)) * width;
      const y = height - ((v - min) / range) * (height - 10) - 5;
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");

  const latest = numericValues[numericValues.length - 1];

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-sm text-slate-500">{title}</p>
        <p className="text-sm font-semibold text-slate-900">{latest}</p>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="h-24 w-full">
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          points={points}
          className={colorClass}
        />
      </svg>
    </div>
  );
}

export default function PredictionDashboardPage() {
  const [patientId, setPatientId] = useState("");
  const [soundEnabled, setSoundEnabled] = useState(false);

  const [dashboardData, setDashboardData] = useState<DashboardResponse | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<PatientCard | null>(null);
  const [timeseries, setTimeseries] = useState<TimeSeriesResponse | null>(null);

  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [loadingManual, setLoadingManual] = useState(false);
  const [loadingCharts, setLoadingCharts] = useState(false);
  const [error, setError] = useState("");

  const previousHighCount = useRef(0);

  const levelBadge = (level?: string) => {
    if (level === "High") return "bg-red-100 text-red-700";
    if (level === "Moderate") return "bg-amber-100 text-amber-700";
    if (level === "Low") return "bg-emerald-100 text-emerald-700";
    return "bg-slate-100 text-slate-700";
  };

  const alertBannerClass = (level?: string) => {
    if (level === "High") return "bg-red-600 text-white";
    if (level === "Moderate") return "bg-amber-500 text-white";
    if (level === "Low") return "bg-emerald-600 text-white";
    return "bg-slate-700 text-white";
  };

  const playAlertSound = () => {
    try {
      const AudioContextClass =
        window.AudioContext ||
        (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

      if (!AudioContextClass) return;

      const context = new AudioContextClass();
      const scheduleBeep = (start: number, freq: number, duration: number) => {
        const oscillator = context.createOscillator();
        const gainNode = context.createGain();

        oscillator.type = "square";
        oscillator.frequency.setValueAtTime(freq, start);
        gainNode.gain.setValueAtTime(0.0001, start);
        gainNode.gain.exponentialRampToValueAtTime(0.35, start + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, start + duration);

        oscillator.connect(gainNode);
        gainNode.connect(context.destination);

        oscillator.start(start);
        oscillator.stop(start + duration);
      };

      const now = context.currentTime;
      scheduleBeep(now, 900, 0.18);
      scheduleBeep(now + 0.24, 900, 0.18);
      scheduleBeep(now + 0.48, 700, 0.28);
    } catch (err) {
      console.error("Audio alert failed:", err);
    }
  };

  const fetchDashboard = async () => {
    try {
      setError("");
      const response = await fetch("http://127.0.0.1:8000/dashboard/alerts?limit=8");

      if (!response.ok) {
        throw new Error("Failed to fetch dashboard data");
      }

      const result: DashboardResponse = await response.json();
      setDashboardData(result);

      if (result.patients.length > 0) {
        const patientToSelect =
          selectedPatient &&
          result.patients.find((p) => p.patient_id === selectedPatient.patient_id)
            ? result.patients.find((p) => p.patient_id === selectedPatient.patient_id)!
            : result.patients[0];

        setSelectedPatient(patientToSelect);
        fetchTimeseries(patientToSelect.patient_id);
      }

      if (
        soundEnabled &&
        result.high_risk_count > 0 &&
        result.high_risk_count > previousHighCount.current
      ) {
        playAlertSound();
      }

      previousHighCount.current = result.high_risk_count;
    } catch (err) {
      setError("Could not load dashboard data from FastAPI.");
      console.error(err);
    } finally {
      setLoadingDashboard(false);
    }
  };

  const fetchTimeseries = async (id: number) => {
    try {
      setLoadingCharts(true);
      const response = await fetch(`http://127.0.0.1:8000/patient/${id}/timeseries?max_points=24`);

      if (!response.ok) {
        throw new Error("Failed to fetch patient charts");
      }

      const data: TimeSeriesResponse = await response.json();
      setTimeseries(data);
    } catch (err) {
      console.error(err);
      setTimeseries(null);
    } finally {
      setLoadingCharts(false);
    }
  };

  const handleManualPredict = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!patientId) return;

    try {
      setLoadingManual(true);
      setError("");

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_id: Number(patientId),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch patient prediction");
      }

      const data = await response.json();
      setSelectedPatient(data);
      fetchTimeseries(Number(patientId));

      if (soundEnabled && data.risk_level === "High") {
        playAlertSound();
      }
    } catch (err) {
      setError("Could not fetch patient prediction.");
      console.error(err);
    } finally {
      setLoadingManual(false);
    }
  };

  useEffect(() => {
    setLoadingDashboard(true);
    fetchDashboard();

    const interval = setInterval(() => {
      fetchDashboard();
    }, 20000);

    return () => clearInterval(interval);
  }, [soundEnabled]);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">ICU Early Warning System</h1>
            <p className="text-xs text-slate-500">Clinical dashboard and patient alerts</p>
          </div>

          <div className="flex items-center gap-7 text-sm font-medium">
            <Link href="/" className="text-slate-700 transition hover:text-blue-700">
              Home
            </Link>
            <Link href="/prediction" className="text-slate-700 transition hover:text-blue-700">
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <section className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-6 rounded-[2rem] border border-slate-200 bg-white p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-3xl font-bold">ICU Monitoring Dashboard</h2>
              <p className="mt-2 text-slate-600">
                Search by Patient ID or review the live list of monitored patients.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => setSoundEnabled((prev) => !prev)}
                className={`rounded-2xl px-4 py-3 font-semibold transition ${
                  soundEnabled
                    ? "bg-red-600 text-white hover:bg-red-700"
                    : "bg-slate-100 text-slate-800 hover:bg-slate-200"
                }`}
              >
                {soundEnabled ? "Sound ON" : "Enable Sound"}
              </button>

              <button
                onClick={playAlertSound}
                className="rounded-2xl bg-blue-700 px-4 py-3 font-semibold text-white transition hover:bg-blue-800"
              >
                Test Sound
              </button>

              <button
                onClick={fetchDashboard}
                className="rounded-2xl border border-slate-300 bg-white px-4 py-3 font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        <div className="mb-6 grid gap-4 md:grid-cols-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-sm text-slate-500">Total Patients</p>
            <p className="mt-1 text-3xl font-bold">{dashboardData?.total_patients ?? "-"}</p>
          </div>

          <div className="rounded-2xl border border-red-100 bg-red-50 p-5">
            <p className="text-sm text-red-600">Active Alerts</p>
            <p className="mt-1 text-3xl font-bold text-red-700">
              {dashboardData?.active_alerts ?? "-"}
            </p>
          </div>

          <div className="rounded-2xl border border-amber-100 bg-amber-50 p-5">
            <p className="text-sm text-amber-600">Moderate Risk</p>
            <p className="mt-1 text-3xl font-bold text-amber-700">
              {dashboardData?.moderate_risk_count ?? "-"}
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-5">
            <p className="text-sm text-slate-500">Last Update</p>
            <p className="mt-1 text-lg font-bold text-slate-900">
              {dashboardData?.last_updated_utc
                ? new Date(dashboardData.last_updated_utc).toLocaleTimeString()
                : "-"}
            </p>
          </div>
        </div>

        {error ? (
          <div className="mb-6 rounded-2xl bg-red-50 p-4 text-red-600">{error}</div>
        ) : null}

        <div className="grid gap-8 xl:grid-cols-[0.85fr,1.15fr]">
          <div className="space-y-6">
            <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
              <h3 className="mb-4 text-2xl font-bold">Patient Search</h3>

              <form onSubmit={handleManualPredict} className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">
                    Patient ID
                  </label>
                  <input
                    type="number"
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    className="w-full rounded-2xl border border-slate-300 bg-slate-50 px-4 py-3 outline-none focus:border-blue-500"
                    placeholder="e.g. 2827"
                    required
                  />
                </div>

                <button
                  type="submit"
                  disabled={loadingManual}
                  className="w-full rounded-2xl bg-blue-700 px-4 py-3 font-semibold text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-blue-400"
                >
                  {loadingManual ? "Checking..." : "Check Patient"}
                </button>
              </form>
            </div>

            <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
              <div className="mb-4">
                <h3 className="text-2xl font-bold">Patient List</h3>
                <p className="text-slate-600">
                  Live monitored patients ranked by current alert severity.
                </p>
              </div>

              {loadingDashboard ? (
                <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-500">
                  Loading patients...
                </div>
              ) : dashboardData?.patients?.length ? (
                <div className="space-y-3">
                  {dashboardData.patients.map((patient) => (
                    <button
                      key={patient.patient_id}
                      onClick={() => {
                        setSelectedPatient(patient);
                        fetchTimeseries(patient.patient_id);
                      }}
                      className={`w-full rounded-2xl border p-4 text-left transition ${
                        selectedPatient?.patient_id === patient.patient_id
                          ? "border-blue-500 bg-blue-50"
                          : "border-slate-200 bg-white hover:border-blue-300 hover:bg-slate-50"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <p className="text-sm text-slate-500">Patient ID</p>
                          <p className="text-xl font-bold">{patient.patient_id}</p>
                        </div>

                        <span
                          className={`rounded-full px-3 py-1 text-sm font-semibold ${levelBadge(
                            patient.risk_level
                          )}`}
                        >
                          {patient.risk_level}
                        </span>
                      </div>

                      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                        <div className="rounded-xl bg-slate-50 p-3">
                          <p className="text-slate-500">Risk Score</p>
                          <p className="font-bold">{patient.risk_score}</p>
                        </div>

                        <div className="rounded-xl bg-slate-50 p-3">
                          <p className="text-slate-500">Lead Time</p>
                          <p className="font-bold">{patient.lead_time_hours}h</p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-500">
                  No patients available.
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
              <div className="mb-5">
                <h3 className="text-2xl font-bold">Patient Details</h3>
                <p className="text-slate-600">
                  Clinical status, latest vital signs, and recommended action.
                </p>
              </div>

              {selectedPatient ? (
                <div className="space-y-5">
                  {selectedPatient.alert_banner && (
                    <div
                      className={`rounded-[1.5rem] p-5 text-center text-lg font-bold ${alertBannerClass(
                        selectedPatient.risk_level
                      )}`}
                    >
                      {selectedPatient.alert_banner}
                    </div>
                  )}

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl bg-slate-100 p-4">
                      <p className="text-sm text-slate-500">Patient ID</p>
                      <p className="mt-1 text-2xl font-bold">{selectedPatient.patient_id}</p>
                    </div>

                    <div className="rounded-2xl bg-slate-100 p-4">
                      <p className="text-sm text-slate-500">Risk Level</p>
                      <p className="mt-1 text-2xl font-bold">{selectedPatient.risk_level}</p>
                    </div>

                    <div className="rounded-2xl bg-blue-50 p-4">
                      <p className="text-sm text-slate-500">Risk Score</p>
                      <p className="mt-1 text-2xl font-bold text-blue-700">
                        {selectedPatient.risk_score}
                      </p>
                    </div>

                    <div className="rounded-2xl bg-emerald-50 p-4">
                      <p className="text-sm text-slate-500">Lead Time</p>
                      <p className="mt-1 text-2xl font-bold text-emerald-700">
                        {selectedPatient.lead_time_hours} hours
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl bg-slate-50 p-4 ring-1 ring-slate-200">
                    <p className="text-sm text-slate-500">Recommended Action</p>
                    <p className="mt-1 text-lg font-semibold text-slate-900">
                      {selectedPatient.recommendation ?? "Clinical review recommended."}
                    </p>
                  </div>

                  {selectedPatient.latest_vitals && (
                    <div className="rounded-[1.5rem] bg-slate-50 p-5">
                      <h4 className="mb-4 text-lg font-bold">Latest Vital Signs</h4>

                      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">Heart Rate</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.heart_rate ?? "N/A"}
                          </p>
                        </div>

                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">SpO2</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.spo2 ?? "N/A"}
                          </p>
                        </div>

                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">Respiratory Rate</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.respiratory_rate ?? "N/A"}
                          </p>
                        </div>

                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">Systolic BP</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.systolic_bp ?? "N/A"}
                          </p>
                        </div>

                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">Lactate</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.lactate ?? "N/A"}
                          </p>
                        </div>

                        <div className="rounded-2xl bg-white p-4 ring-1 ring-slate-200">
                          <p className="text-sm text-slate-500">Latest Hour</p>
                          <p className="mt-1 text-xl font-bold">
                            {selectedPatient.latest_vitals.latest_hour_from_panel ?? "N/A"}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="rounded-2xl bg-slate-50 p-8 text-center text-slate-500">
                  Choose a patient from the list or search by Patient ID.
                </div>
              )}
            </div>

            <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
              <div className="mb-5">
                <h3 className="text-2xl font-bold">Vital Sign Trends</h3>
                <p className="text-slate-600">
                  Recent measurements for the selected patient.
                </p>
              </div>

              {loadingCharts ? (
                <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-500">
                  Loading charts...
                </div>
              ) : timeseries ? (
                <div className="grid gap-4 md:grid-cols-2">
                  <SparklineCard
                    title="Heart Rate"
                    values={timeseries.heart_rate}
                    colorClass="text-blue-700"
                  />
                  <SparklineCard
                    title="Respiratory Rate"
                    values={timeseries.respiratory_rate}
                    colorClass="text-emerald-700"
                  />
                  <SparklineCard
                    title="Systolic BP"
                    values={timeseries.systolic_bp}
                    colorClass="text-violet-700"
                  />
                  <SparklineCard
                    title="SpO2"
                    values={timeseries.spo2}
                    colorClass="text-cyan-700"
                  />
                  <SparklineCard
                    title="Lactate"
                    values={timeseries.lactate}
                    colorClass="text-red-700"
                  />
                </div>
              ) : (
                <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-500">
                  No charts available.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}