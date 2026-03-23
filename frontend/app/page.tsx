import Link from "next/link";

const CONTACT_EMAIL = "meriemdokkar9@gmail.com";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">ICU Early Warning System</h1>
            <p className="text-xs text-slate-500">Clinical monitoring and early alerts</p>
          </div>

          <div className="hidden items-center gap-7 text-sm font-medium md:flex">
            <a href="#home" className="text-slate-700 transition hover:text-blue-700">
              Home
            </a>
            <Link href="/prediction" className="text-slate-700 transition hover:text-blue-700">
              Dashboard
            </Link>
            <a href="#about" className="text-slate-700 transition hover:text-blue-700">
              About
            </a>
            <a href="#results" className="text-slate-700 transition hover:text-blue-700">
              Results
            </a>
            <a href="#contact" className="text-slate-700 transition hover:text-blue-700">
              Contact
            </a>
          </div>
        </div>
      </nav>

      <section
        id="home"
        className="mx-auto grid max-w-7xl items-center gap-10 px-6 py-14 md:grid-cols-2 md:py-20"
      >
        <div className="space-y-6">
          <span className="inline-flex rounded-full bg-blue-50 px-4 py-1 text-sm font-semibold text-blue-700 ring-1 ring-blue-100">
            Intensive Care Monitoring Platform
          </span>

          <h2 className="text-4xl font-bold leading-tight md:text-6xl">
            ICU Early Warning
            <span className="block text-blue-700">for Patient Deterioration</span>
          </h2>

          <p className="max-w-xl text-lg leading-8 text-slate-600">
            A hospital-oriented system that monitors patient status, retrieves vital
            signs automatically, and generates clear early alerts to support
            clinical decision-making in intensive care units.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link
              href="/prediction"
              className="rounded-2xl bg-blue-700 px-6 py-3 font-semibold text-white shadow-sm transition hover:bg-blue-800"
            >
              Open Monitoring Dashboard
            </Link>

            <a
              href="#about"
              className="rounded-2xl border border-slate-300 bg-white px-6 py-3 font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
            >
              Learn More
            </a>
          </div>
        </div>

        <div className="relative">
          <div className="overflow-hidden rounded-[2rem] border border-slate-200 bg-white shadow-xl">
            <img
              src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzSHaMmcQaKb0FcxdsyvkqElsRxR3ceQ6cdw&s"
              alt="ICU hospital room"
              className="h-[540px] w-full object-cover"
            />
          </div>
        </div>
      </section>

      <section id="about" className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-8">
          <h3 className="text-3xl font-bold">About</h3>
          <p className="mt-3 max-w-3xl text-slate-600">
            This platform is designed for clinical use. It focuses on patient status,
            vital signs, alert visibility, and timely medical attention rather than
            technical AI details.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-3xl border border-slate-200 bg-white p-6">
            <h4 className="mb-2 text-xl font-semibold">Live Monitoring</h4>
            <p className="text-slate-600">
              View current patient conditions and alerts in a clean hospital-style dashboard.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6">
            <h4 className="mb-2 text-xl font-semibold">Automatic Data Retrieval</h4>
            <p className="text-slate-600">
              The system fetches the latest measurements directly using the patient ID.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6">
            <h4 className="mb-2 text-xl font-semibold">Early Alerts</h4>
            <p className="text-slate-600">
              Clear visual and audio alerts help clinicians identify at-risk patients earlier.
            </p>
          </div>
        </div>
      </section>

      <section id="results" className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-8">
          <h3 className="text-3xl font-bold">System Focus</h3>
          <p className="mt-3 max-w-3xl text-slate-600">
            The platform is centered on early warning, monitoring clarity, and actionable
            patient information for intensive care environments.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
            <p className="text-sm text-slate-500">Alerting</p>
            <p className="mt-2 text-3xl font-bold text-slate-900">Real-Time</p>
            <p className="mt-2 text-slate-600">Visual and audio notification support.</p>
          </div>

          <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
            <p className="text-sm text-slate-500">Patient Review</p>
            <p className="mt-2 text-3xl font-bold text-slate-900">Fast</p>
            <p className="mt-2 text-slate-600">Clear patient details and latest vital signs.</p>
          </div>

          <div className="rounded-[2rem] border border-slate-200 bg-white p-6">
            <p className="text-sm text-slate-500">Monitoring</p>
            <p className="mt-2 text-3xl font-bold text-slate-900">Continuous</p>
            <p className="mt-2 text-slate-600">Designed for ICU patient oversight.</p>
          </div>
        </div>
      </section>

      <section id="contact" className="mx-auto max-w-7xl px-6 pb-16 pt-8">
        <div className="rounded-[2rem] border border-slate-200 bg-white p-8 md:flex md:items-center md:justify-between">
          <div>
            <h3 className="text-2xl font-bold">Contact</h3>
            <p className="mt-2 text-slate-600">For communication regarding the system:</p>
            <p className="mt-3 text-lg font-semibold text-blue-700">{CONTACT_EMAIL}</p>
          </div>

          <Link
            href="/prediction"
            className="mt-6 inline-flex rounded-2xl bg-blue-700 px-6 py-3 font-semibold text-white transition hover:bg-blue-800 md:mt-0"
          >
            Go to Dashboard
          </Link>
        </div>
      </section>
    </main>
  );
}