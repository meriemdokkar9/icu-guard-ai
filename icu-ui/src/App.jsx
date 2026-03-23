import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bell,
  CheckCircle2,
  Eye,
  EyeOff,
  FileText,
  HeartPulse,
  LockKeyhole,
  Siren,
  Stethoscope,
  UserCircle2,
  Volume2,
  VolumeX,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function App() {
  const [page, setPage] = useState("login");
  const [credentials, setCredentials] = useState({ username: "", password: "" });
  const [loginError, setLoginError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  const [historyItems, setHistoryItems] = useState([]);
  const [alertsItems, setAlertsItems] = useState([]);
  const [demoCases, setDemoCases] = useState([]);
  const [loadingDemoCases, setLoadingDemoCases] = useState(false);
  const [riskFilter, setRiskFilter] = useState("all");

  const [soundEnabled, setSoundEnabled] = useState(true);

  const previousAlertCountRef = useRef(0);
  const inputRefs = useRef([]);

  const demoAccount = {
    username: "doctor@hospital.org",
    password: "Admin123!",
    displayName: "Dr. Meriem",
    role: "ICU Physician",
  };

  const [formData, setFormData] = useState({
    patientId: "",
    hourFromAdmission: "",
    age: "",
    heartRate: "",
    bloodPressure: "",
    respiratoryRate: "",
    temperature: "",
    spo2: "",
    lactate: "",
    creatinine: "",
    gcs: "",
  });

  const displayedRisk = predictionResult?.risk_category || "Low";
  const displayedScore = predictionResult?.risk_score ?? "-";
  const displayedConfidence =
    predictionResult?.confidence !== undefined && predictionResult?.confidence !== null
      ? `${Math.round(predictionResult.confidence * 100)}%`
      : "-";
  const displayedFactors = predictionResult?.top_factors || [];
  const clinicalMessage = predictionResult?.clinical_message || null;

  const riskBadge = (risk) => {
    const normalized = String(risk || "").toLowerCase();
    if (normalized === "high") return "bg-red-50 text-red-700 ring-red-200";
    if (normalized === "medium") return "bg-amber-50 text-amber-700 ring-amber-200";
    return "bg-emerald-50 text-emerald-700 ring-emerald-200";
  };

  const scoreText = (value) => {
    if (value === null || value === undefined || value === "") return "-";
    const num = Number(value);
    if (Number.isNaN(num)) return String(value);
    return num.toFixed(4);
  };

  const playAlertTone = () => {
    try {
      if (!soundEnabled) return;
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      gainNode.gain.setValueAtTime(0.001, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.12, audioContext.currentTime + 0.02);
      gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.35);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.35);
    } catch (error) {
      console.error(error);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/history`);
      const data = await response.json();
      setHistoryItems(data.items || []);
    } catch (error) {
      console.error(error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE}/alerts`);
      const data = await response.json();
      const items = data.items || [];
      setAlertsItems(items);

      if (items.length > previousAlertCountRef.current && previousAlertCountRef.current > 0) {
        playAlertTone();
      }
      previousAlertCountRef.current = items.length;
    } catch (error) {
      console.error(error);
    }
  };

  const fetchDemoCases = async (selectedRisk = riskFilter) => {
    try {
      setLoadingDemoCases(true);
      const response = await fetch(`${API_BASE}/demo-cases?limit=50&risk=${selectedRisk}`);
      const data = await response.json();
      setDemoCases(data.items || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoadingDemoCases(false);
    }
  };

  const acknowledgeAlert = async (alertId) => {
    try {
      await fetch(`${API_BASE}/alerts/${alertId}/acknowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewed_by: demoAccount.displayName }),
      });
      fetchAlerts();
    } catch (error) {
      console.error(error);
    }
  };

  const focusFirstInput = () => {
    setTimeout(() => {
      const first = inputRefs.current[0];
      if (first) {
        first.focus();
        first.select?.();
      }
    }, 80);
  };

  const clearForm = () => {
    setFormData({
      patientId: "",
      hourFromAdmission: "",
      age: "",
      heartRate: "",
      bloodPressure: "",
      respiratoryRate: "",
      temperature: "",
      spo2: "",
      lactate: "",
      creatinine: "",
      gcs: "",
    });
    focusFirstInput();
  };

  const loadCaseIntoForm = (item) => {
    setFormData({
      patientId: String(item.patient_id),
      hourFromAdmission: String(item.hour_from_admission),
      age: String(item.age),
      heartRate: String(item.heart_rate),
      bloodPressure: `${item.systolic_bp}/${item.diastolic_bp}`,
      respiratoryRate: String(item.respiratory_rate),
      temperature: String(item.temperature_c),
      spo2: String(item.spo2_pct),
      lactate: String(item.lactate),
      creatinine: String(item.creatinine),
      gcs: "11",
    });
    setPage("new");
    focusFirstInput();
  };

  const predictDemoCase = async (item) => {
    try {
      setLoadingPrediction(true);
      setPredictionError("");

      const url = `${API_BASE}/predict-demo?patient_id=${encodeURIComponent(
        item.patient_id
      )}&hour_from_admission=${encodeURIComponent(item.hour_from_admission)}`;

      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed demo prediction");

      const data = await response.json();
      setPredictionResult(data);

      if (data.input_snapshot) {
        setFormData({
          patientId: String(data.patient_id),
          hourFromAdmission: String(data.hour_from_admission),
          age: String(data.input_snapshot.age),
          heartRate: String(data.input_snapshot.heart_rate),
          bloodPressure: `${data.input_snapshot.systolic_bp}/${data.input_snapshot.diastolic_bp}`,
          respiratoryRate: String(data.input_snapshot.respiratory_rate),
          temperature: String(data.input_snapshot.temperature_c),
          spo2: String(data.input_snapshot.spo2_pct),
          lactate: String(data.input_snapshot.lactate),
          creatinine: String(data.input_snapshot.creatinine),
          gcs: "11",
        });
      }

      setPage("result");
      fetchHistory();
      fetchAlerts();

      if (data.alert_created) playAlertTone();
    } catch (error) {
      console.error(error);
      setPredictionError("Demo case prediction failed.");
    } finally {
      setLoadingPrediction(false);
    }
  };

  const runPrediction = async () => {
    try {
      setLoadingPrediction(true);
      setPredictionError("");

      const bp = (formData.bloodPressure || "0/0").split("/");
      const systolic = Number(bp[0] || 0);
      const diastolic = Number(bp[1] || 0);

      const payload = {
        patient_id: formData.patientId || "ICU-2408",
        hour_from_admission: Number(formData.hourFromAdmission || 24),
        heart_rate: Number(formData.heartRate),
        respiratory_rate: Number(formData.respiratoryRate),
        spo2_pct: Number(formData.spo2),
        temperature_c: Number(formData.temperature),
        systolic_bp: systolic,
        diastolic_bp: diastolic,
        oxygen_device: "Nasal Cannula",
        oxygen_flow: 2.0,
        mobility_score: 2.0,
        nurse_alert: 1.0,
        wbc_count: 12.4,
        lactate: Number(formData.lactate),
        creatinine: Number(formData.creatinine),
        crp_level: 45.0,
        hemoglobin: 11.2,
        sepsis_risk_score: 0.72,
        age: Number(formData.age),
        gender: "Female",
        comorbidity_index: 3.0,
        admission_type: "Emergency",
        baseline_risk_score: 0.55,
        los_hours: 24.0,
      };

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setPredictionResult(data);
      setPage("result");
      fetchHistory();
      fetchAlerts();

      if (data.alert_created) playAlertTone();
    } catch (error) {
      console.error(error);
      setPredictionError("Prediction failed. Make sure FastAPI is running.");
    } finally {
      setLoadingPrediction(false);
    }
  };

  const handleFieldKeyDown = (e, index) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const next = inputRefs.current[index + 1];
      if (next) {
        next.focus();
        next.select?.();
      } else {
        runPrediction();
      }
    }
  };

  useEffect(() => {
    if (page !== "login") {
      fetchHistory();
      fetchAlerts();
      fetchDemoCases(riskFilter);
      const interval = setInterval(() => {
        fetchAlerts();
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [page]);

  useEffect(() => {
    if (page === "new") focusFirstInput();
  }, [page]);

  const card = "rounded-3xl border border-slate-200 bg-white p-6 shadow-sm";
  const inputClass =
    "w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-slate-900 focus:ring-4 focus:ring-slate-100";

  const highCount = useMemo(
    () => demoCases.filter((x) => String(x.risk_category).toLowerCase() === "high").length,
    [demoCases]
  );

  const mediumCount = useMemo(
    () => demoCases.filter((x) => String(x.risk_category).toLowerCase() === "medium").length,
    [demoCases]
  );

  const LoginPage = () => (
    <div className="min-h-screen bg-slate-100 px-6 py-8">
      <div className="mx-auto grid min-h-[90vh] max-w-7xl items-center gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="relative overflow-hidden rounded-[36px] border bg-slate-900 shadow-xl">
          <img
            src="https://images.unsplash.com/photo-1586773860418-d37222d8fce3?auto=format&fit=crop&w=1600&q=80"
            alt="ICU"
            className="absolute inset-0 h-full w-full object-cover opacity-35"
          />
          <div className="absolute inset-0 bg-gradient-to-br from-slate-950/80 via-slate-900/70 to-cyan-950/45" />
          <div className="relative z-10 flex h-full flex-col justify-between p-8 text-white lg:p-10">
            <div className="inline-flex w-fit rounded-full border border-white/15 bg-white/10 px-4 py-2 text-sm">
              Hospital Internal Platform
            </div>
            <div>
              <p className="text-sm uppercase tracking-[0.28em] text-cyan-200/80">ICU Guard AI</p>
              <h1 className="mt-4 text-5xl font-semibold tracking-tight lg:text-6xl">
                Early deterioration insights for intensive care teams.
              </h1>
              <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-200">
                Clinical dashboard, live alerts, exact demo cases, and physician-facing decision support.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-[36px] border border-slate-200 bg-white p-8 shadow-xl lg:p-10">
          <div className="mb-8">
            <p className="text-sm uppercase tracking-[0.18em] text-slate-400">Secure Sign In</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900">Access the platform</h2>
          </div>

          <div className="mb-6 rounded-3xl border border-slate-200 bg-slate-50 p-5 text-sm text-slate-700">
            <p className="font-semibold text-slate-900">Demo account</p>
            <p className="mt-2">Username: {demoAccount.username}</p>
            <p>Password: {demoAccount.password}</p>
            <button
              type="button"
              onClick={() => {
                setCredentials({ username: demoAccount.username, password: demoAccount.password });
                setLoginError("");
              }}
              className="mt-3 rounded-2xl bg-white px-4 py-2 font-medium text-slate-900 shadow-sm ring-1 ring-slate-200 hover:bg-slate-100"
            >
              Use Demo Account
            </button>
          </div>

          <div className="space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">Email or Username</span>
              <div className="flex items-center rounded-2xl border border-slate-200 px-4 py-4">
                <UserCircle2 className="mr-3 h-5 w-5 text-slate-400" />
                <input
                  value={credentials.username}
                  onChange={(e) => {
                    setCredentials((prev) => ({ ...prev, username: e.target.value }));
                    setLoginError("");
                  }}
                  className="w-full bg-transparent outline-none"
                  placeholder="doctor@hospital.org"
                />
              </div>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-700">Password</span>
              <div className="flex items-center rounded-2xl border border-slate-200 px-4 py-4">
                <LockKeyhole className="mr-3 h-5 w-5 text-slate-400" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={credentials.password}
                  onChange={(e) => {
                    setCredentials((prev) => ({ ...prev, password: e.target.value }));
                    setLoginError("");
                  }}
                  className="w-full bg-transparent outline-none"
                  placeholder="••••••••"
                />
                <button type="button" onClick={() => setShowPassword((prev) => !prev)}>
                  {showPassword ? <EyeOff className="h-5 w-5 text-slate-500" /> : <Eye className="h-5 w-5 text-slate-500" />}
                </button>
              </div>
            </label>

            {loginError ? <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700">{loginError}</div> : null}

            <button
              onClick={() => {
                if (
                  credentials.username.trim().toLowerCase() === demoAccount.username.toLowerCase() &&
                  credentials.password.trim() === demoAccount.password
                ) {
                  setPage("dashboard");
                  fetchHistory();
                  fetchAlerts();
                  fetchDemoCases(riskFilter);
                } else {
                  setLoginError("Invalid username or password.");
                }
              }}
              className="flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-900 px-5 py-4 text-base font-semibold text-white hover:bg-slate-800"
            >
              Sign In
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const Layout = ({ title, subtitle, children }) => (
    <div className="min-h-screen bg-[#f4f7fb] text-slate-900">
      <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-900 text-white">
              <HeartPulse className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Clinical AI Platform</p>
              <h1 className="text-lg font-semibold tracking-tight text-slate-900">ICU Guard AI</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSoundEnabled((prev) => !prev)}
              className="rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm hover:bg-slate-50"
            >
              <span className="flex items-center gap-2">
                {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                {soundEnabled ? "Sound On" : "Sound Off"}
              </span>
            </button>
            <button
              onClick={() => setPage("alerts")}
              className="relative rounded-2xl border border-slate-200 bg-white p-3 hover:bg-slate-50"
            >
              <Bell className="h-4 w-4" />
              {alertsItems.length > 0 ? (
                <span className="absolute -right-1 -top-1 flex h-5 min-w-5 items-center justify-center rounded-full bg-red-600 px-1 text-[10px] font-semibold text-white">
                  {alertsItems.length}
                </span>
              ) : null}
            </button>
            <button
              onClick={() => setPage("login")}
              className="rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm hover:bg-slate-50"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl grid-cols-12 gap-6 px-6 py-6">
        <aside className="col-span-12 lg:col-span-3 xl:col-span-2">
          <div className="sticky top-24 space-y-4 rounded-[30px] border border-slate-200 bg-white p-4 shadow-sm">
            <div className="rounded-3xl bg-gradient-to-br from-slate-900 to-slate-800 p-5 text-white">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-300">Authorized access</p>
              <p className="mt-3 text-lg font-semibold">{demoAccount.displayName}</p>
              <p className="mt-1 text-sm text-slate-300">{demoAccount.role}</p>
            </div>

            {[
              ["dashboard", "Dashboard", Activity],
              ["alerts", "Critical Alerts", Siren],
              ["new", "New Prediction", Stethoscope],
              ["result", "Prediction Result", AlertTriangle],
              ["history", "Prediction History", FileText],
            ].map(([key, label, Icon]) => (
              <button
                key={key}
                onClick={() => {
                  setPage(key);
                  if (key === "alerts") fetchAlerts();
                  if (key === "history") fetchHistory();
                  if (key === "new") focusFirstInput();
                }}
                className={`flex w-full items-center justify-between rounded-2xl px-4 py-3 text-left text-sm font-medium ${
                  page === key ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                <span className="flex items-center gap-3">
                  <Icon className="h-4 w-4" />
                  {label}
                </span>
                {key === "alerts" && alertsItems.length > 0 ? (
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${page === key ? "bg-white text-slate-900" : "bg-red-100 text-red-700"}`}>
                    {alertsItems.length}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
        </aside>

        <main className="col-span-12 space-y-6 lg:col-span-9 xl:col-span-10">
          <div className={card}>
            <h2 className="text-3xl font-semibold tracking-tight text-slate-900">{title}</h2>
            <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
          </div>
          {children}
        </main>
      </div>
    </div>
  );

  const DashboardPage = () => (
    <Layout title="Dashboard" subtitle="Exact fusion cases from the real dataset, sorted by clinical priority.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ["Cases Loaded", String(demoCases.length || 0)],
          ["High Risk", String(highCount)],
          ["Medium Risk", String(mediumCount)],
          ["Active Alerts", String(alertsItems.length || 0)],
        ].map(([label, value]) => (
          <div key={label} className={card}>
            <p className="text-sm text-slate-500">{label}</p>
            <p className="mt-2 text-3xl font-semibold text-slate-900">{value}</p>
          </div>
        ))}
      </div>

      <div className={card}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-xl font-semibold text-slate-900">Top cases from training data</h3>
            <p className="text-sm text-slate-500">These rows use exact XGBoost + GRU + Fusion from the original dataset.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              ["all", "All"],
              ["high", "High"],
              ["medium", "Medium"],
              ["low", "Low"],
            ].map(([value, label]) => (
              <button
                key={value}
                onClick={() => {
                  setRiskFilter(value);
                  fetchDemoCases(value);
                }}
                className={`rounded-2xl px-4 py-2 text-sm font-medium ${
                  riskFilter === value
                    ? "bg-slate-900 text-white"
                    : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                }`}
              >
                {label}
              </button>
            ))}
            <button
              onClick={() => fetchDemoCases(riskFilter)}
              className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              {loadingDemoCases ? "Loading..." : "Reload Cases"}
            </button>
          </div>
        </div>

        <div className="overflow-auto rounded-3xl border border-slate-200">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">Patient ID</th>
                <th className="px-4 py-3 font-medium">Hour</th>
                <th className="px-4 py-3 font-medium">Risk</th>
                <th className="px-4 py-3 font-medium">Final Score</th>
                <th className="px-4 py-3 font-medium">XGB</th>
                <th className="px-4 py-3 font-medium">GRU</th>
                <th className="px-4 py-3 font-medium">Fusion</th>
                <th className="px-4 py-3 font-medium">Mode</th>
                <th className="px-4 py-3 font-medium">True Label</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {demoCases.map((item, index) => (
                <tr key={`${item.patient_id}-${item.hour_from_admission}-${index}`} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-3 font-medium text-slate-900">{item.patient_id}</td>
                  <td className="px-4 py-3 text-slate-600">{item.hour_from_admission}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ring-1 ${riskBadge(item.risk_category)}`}>
                      {item.risk_category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.risk_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.xgb_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.gru_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.fusion_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{item.model_mode}</td>
                  <td className="px-4 py-3 text-slate-700">{item.true_label}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => predictDemoCase(item)}
                        className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-medium text-white hover:bg-slate-800"
                      >
                        Predict Now
                      </button>
                      <button
                        onClick={() => loadCaseIntoForm(item)}
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
                      >
                        Load in Form
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {demoCases.length === 0 ? (
                <tr>
                  <td colSpan="10" className="px-4 py-10 text-center text-slate-500">
                    No cases loaded.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  );

  const NewPredictionPage = () => {
    const fields = [
      ["patientId", "Patient ID"],
      ["hourFromAdmission", "Hour from Admission"],
      ["age", "Age"],
      ["heartRate", "Heart Rate"],
      ["bloodPressure", "Blood Pressure"],
      ["respiratoryRate", "Respiratory Rate"],
      ["temperature", "Temperature"],
      ["spo2", "SpO2"],
      ["lactate", "Lactate"],
      ["creatinine", "Creatinine"],
      ["gcs", "GCS"],
    ];

    return (
      <Layout title="New Prediction" subtitle="The exact version: enter patient ID and hour if you want exact 24h Fusion from dataset.">
        <div className="grid gap-6 xl:grid-cols-3">
          <div className={`${card} xl:col-span-2`}>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                runPrediction();
              }}
            >
              <div className="grid gap-4 md:grid-cols-2">
                {fields.map(([key, label], index) => (
                  <label key={key} className="block">
                    <span className="mb-2 block text-sm font-medium text-slate-700">{label}</span>
                    <input
                      ref={(el) => (inputRefs.current[index] = el)}
                      value={formData[key]}
                      onChange={(e) => setFormData((prev) => ({ ...prev, [key]: e.target.value }))}
                      onKeyDown={(e) => handleFieldKeyDown(e, index)}
                      placeholder={`Enter ${label}`}
                      className={inputClass}
                      autoComplete="off"
                    />
                  </label>
                ))}
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                <button
                  type="submit"
                  className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-medium text-white hover:bg-slate-800"
                >
                  {loadingPrediction ? "Predicting..." : "Predict"}
                </button>
                <button
                  type="button"
                  onClick={clearForm}
                  className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Clear Form
                </button>
              </div>
            </form>

            {predictionError ? (
              <div className="mt-4 rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700">
                {predictionError}
              </div>
            ) : null}
          </div>

          <div className="space-y-6">
            <div className={card}>
              <h3 className="text-lg font-semibold text-slate-900">Exact Fusion rule</h3>
              <div className="mt-3 space-y-3 text-sm text-slate-600">
                <p>If patient ID and hour match a real dataset row, the system uses exact 24h Fusion.</p>
                <p>If no exact match is found, it falls back to XGBoost snapshot only.</p>
                <p>The first field activates automatically, and Enter moves forward.</p>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  };

  const ResultPage = () => (
    <Layout title="Prediction Result" subtitle="Exact Fusion when dataset match exists, otherwise XGBoost fallback.">
      <div className="grid gap-6 xl:grid-cols-3">
        <div className={`${card} xl:col-span-2`}>
          <div className={`rounded-3xl border p-6 ${displayedRisk === "High" ? "border-red-200 bg-red-50" : displayedRisk === "Medium" ? "border-amber-200 bg-amber-50" : "border-emerald-200 bg-emerald-50"}`}>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-sm text-slate-500">Patient ID</p>
                <h3 className="mt-1 text-3xl font-semibold text-slate-900">{predictionResult?.patient_id || "-"}</h3>
                <p className="mt-1 text-sm text-slate-500">Created at {predictionResult?.created_at || "-"}</p>
              </div>
              <span className={`inline-flex rounded-full px-4 py-2 text-sm font-semibold ring-1 ${riskBadge(displayedRisk)}`}>
                {displayedRisk} Risk
              </span>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">Final Risk Score</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{scoreText(displayedScore)}</p>
              </div>
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">Confidence</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{displayedConfidence}</p>
              </div>
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">Alert Created</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">
                  {predictionResult?.alert_created ? "Yes" : "No"}
                </p>
              </div>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">XGB Score</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">{scoreText(predictionResult?.xgb_score)}</p>
              </div>
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">GRU Score</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">{scoreText(predictionResult?.gru_score)}</p>
              </div>
              <div className="rounded-2xl bg-white/80 p-4">
                <p className="text-sm text-slate-500">Fusion Score</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">{scoreText(predictionResult?.fusion_score)}</p>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-6">
            <h3 className="text-xl font-semibold text-slate-900">Clinical Support Message</h3>
            <div className="mt-4 rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <p className="text-lg font-semibold text-slate-900">
                {clinicalMessage?.headline || "No clinical message available"}
              </p>
              <p className="mt-3 text-sm leading-7 text-slate-700">
                {clinicalMessage?.summary || "Prediction summary will appear here."}
              </p>
              <div className="mt-4">
                <p className="text-sm font-semibold text-slate-900">Suggested actions</p>
                <div className="mt-3 space-y-2">
                  {(clinicalMessage?.actions || []).map((item, idx) => (
                    <div key={idx} className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700 ring-1 ring-slate-200">
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {predictionResult?.note ? (
            <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-6">
              <h3 className="text-xl font-semibold text-slate-900">Model Note</h3>
              <p className="mt-3 text-sm leading-7 text-slate-700">{predictionResult.note}</p>
            </div>
          ) : null}

          <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-6">
            <h3 className="text-xl font-semibold text-slate-900">Top Contributing Variables</h3>
            <div className="mt-4 grid gap-3">
              {displayedFactors.map((name) => (
                <div key={name} className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3 text-sm">
                  <span className="font-medium text-slate-800">{name}</span>
                  <span className="text-slate-500">important factor</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className={card}>
            <p className="text-sm font-semibold text-slate-700">Actions</p>
            <div className="mt-4 flex flex-col gap-3">
              <button
                onClick={() => {
                  setPage("new");
                  focusFirstInput();
                }}
                className="rounded-2xl bg-slate-900 px-4 py-3 text-sm font-medium text-white hover:bg-slate-800"
              >
                New Prediction
              </button>
              <button
                onClick={() => setPage("alerts")}
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Open Alerts
              </button>
            </div>
          </div>

          <div className={card}>
            <p className="text-sm font-semibold text-slate-700">Source</p>
            <p className="mt-2 text-sm text-slate-600">{predictionResult?.source || "manual_form"}</p>
            <p className="mt-3 text-sm font-semibold text-slate-700">Model Mode</p>
            <p className="mt-2 text-sm text-slate-600">{predictionResult?.model_mode || "-"}</p>
            <p className="mt-3 text-sm font-semibold text-slate-700">Model Used</p>
            <p className="mt-2 text-sm text-slate-600 break-all">{predictionResult?.model_used || "-"}</p>
            <p className="mt-3 text-sm font-semibold text-slate-700">Hour from Admission</p>
            <p className="mt-2 text-sm text-slate-600">{predictionResult?.hour_from_admission ?? "-"}</p>
          </div>
        </div>
      </div>
    </Layout>
  );

  const AlertsPage = () => (
    <Layout title="Critical Alerts" subtitle="Patients requiring review, with suggested clinical action.">
      <div className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
        <div className={card}>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xl font-semibold text-slate-900">Active Alert Queue</h3>
            <button
              onClick={fetchAlerts}
              className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              Refresh
            </button>
          </div>

          {alertsItems.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center text-slate-500">
              No active alerts.
            </div>
          ) : (
            <div className="space-y-4">
              {alertsItems.map((alert) => (
                <div key={alert.alert_id} className="rounded-3xl border border-red-200 bg-red-50 p-5">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ring-1 ${riskBadge(alert.risk_category)}`}>
                          {alert.risk_category}
                        </span>
                        <span className="text-xs font-medium uppercase tracking-[0.15em] text-red-700">
                          {alert.status}
                        </span>
                      </div>
                      <h3 className="mt-3 text-2xl font-semibold text-slate-900">Patient {alert.patient_id}</h3>
                      <p className="mt-1 text-sm text-slate-600">Created at: {alert.created_at}</p>
                    </div>

                    <button
                      onClick={() => acknowledgeAlert(alert.alert_id)}
                      className="rounded-2xl bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-100"
                    >
                      Acknowledge
                    </button>
                  </div>

                  <div className="mt-5 grid gap-3 md:grid-cols-2">
                    <div className="rounded-2xl bg-white/80 p-4">
                      <p className="text-sm text-slate-500">Risk Score</p>
                      <p className="mt-2 text-2xl font-semibold text-slate-900">{scoreText(alert.risk_score)}</p>
                    </div>
                    <div className="rounded-2xl bg-white/80 p-4">
                      <p className="text-sm text-slate-500">Confidence</p>
                      <p className="mt-2 text-2xl font-semibold text-slate-900">
                        {Math.round((alert.confidence || 0) * 100)}%
                      </p>
                    </div>
                  </div>

                  {alert.note ? (
                    <div className="mt-4 rounded-2xl bg-white p-4 ring-1 ring-red-100">
                      <p className="text-sm text-slate-700">{alert.note}</p>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className={card}>
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-5 w-5 text-emerald-600" />
              <h3 className="text-lg font-semibold text-slate-900">Alert Summary</h3>
            </div>
            <div className="mt-4 grid gap-4">
              <div className="rounded-2xl bg-slate-50 p-4">
                <p className="text-sm text-slate-500">Active Alerts</p>
                <p className="mt-2 text-3xl font-semibold text-slate-900">{alertsItems.length}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );

  const HistoryPage = () => (
    <Layout title="Prediction History" subtitle="Saved predictions from manual form and exact demo cases.">
      <div className={card}>
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-xl font-semibold text-slate-900">Saved Prediction History</h3>
          <button
            onClick={fetchHistory}
            className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
          >
            Refresh
          </button>
        </div>

        <div className="overflow-auto rounded-3xl border border-slate-200">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">Patient ID</th>
                <th className="px-4 py-3 font-medium">Hour</th>
                <th className="px-4 py-3 font-medium">Created At</th>
                <th className="px-4 py-3 font-medium">Risk</th>
                <th className="px-4 py-3 font-medium">Final</th>
                <th className="px-4 py-3 font-medium">XGB</th>
                <th className="px-4 py-3 font-medium">GRU</th>
                <th className="px-4 py-3 font-medium">Fusion</th>
                <th className="px-4 py-3 font-medium">Mode</th>
              </tr>
            </thead>
            <tbody>
              {historyItems.map((item, index) => (
                <tr key={`${item.patient_id}-${item.created_at}-${index}`} className="border-t border-slate-100">
                  <td className="px-4 py-3 font-medium text-slate-900">{item.patient_id}</td>
                  <td className="px-4 py-3 text-slate-700">{item.hour_from_admission ?? "-"}</td>
                  <td className="px-4 py-3 text-slate-600">{item.created_at}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ring-1 ${riskBadge(item.risk_category)}`}>
                      {item.risk_category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.risk_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.xgb_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.gru_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{scoreText(item.fusion_score)}</td>
                  <td className="px-4 py-3 text-slate-700">{item.model_mode || "-"}</td>
                </tr>
              ))}
              {historyItems.length === 0 ? (
                <tr>
                  <td colSpan="9" className="px-4 py-10 text-center text-slate-500">
                    No history yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  );

  if (page === "login") return <LoginPage />;
  if (page === "dashboard") return <DashboardPage />;
  if (page === "alerts") return <AlertsPage />;
  if (page === "new") return <NewPredictionPage />;
  if (page === "result") return <ResultPage />;
  if (page === "history") return <HistoryPage />;

  return <DashboardPage />;
}