"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";
import Link from "next/link";

export default function PatientPage() {
  const [demoPatient, setDemoPatient] = useState<Record<string, any> | null>(null);
  const [patients, setPatients] = useState<Record<string, any>[]>([]);
  const [activePatientId, setActivePatientId] = useState<string | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryContext, setSummaryContext] = useState("");
  const [summary, setSummary] = useState<Record<string, any> | null>(null);

  // Lab interpretation state (separate from chat page)
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [labQuestion, setLabQuestion] = useState("");
  const [labResult, setLabResult] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    // Load demo patients for UI (local backend)
    fetch(apiUrl("/v1/demo/patients"))
      .then((r) => r.json())
      .then((list) => {
        setPatients(list);
        if (list.length) {
          setActivePatientId(list[0].id);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!activePatientId) return;
    fetch(apiUrl(`/v1/demo/patients/${activePatientId}`))
      .then((r) => r.json())
      .then(setDemoPatient)
      .catch(() => {});
  }, [activePatientId]);

  // Fallback plan generator (UI-only) to avoid empty plan sections
  const buildFallbackPlan = (demo: any): string[] => {
    if (!demo) return [];
    const problemsText = ((demo.problems || []) as string[]).join(" ").toLowerCase();
    const plan: string[] = [];
    if (!problemsText) return plan;
    plan.push("Continue current medications as prescribed.");
    if (problemsText.includes("diabetes")) {
      plan.push("Monitor blood glucose regularly; follow diet and exercise guidance.");
    }
    if (problemsText.includes("hypertension")) {
      plan.push("Monitor blood pressure at home; reduce sodium and follow lifestyle changes.");
    }
    if (problemsText.includes("asthma")) {
      plan.push("Use rescue inhaler as directed; review triggers and inhaler technique.");
    }
    const labs = demo.last_labs || {};
    if (labs && Object.keys(labs).length) {
      plan.push("Recheck relevant labs as advised.");
    }
    plan.push("Schedule follow-up to reassess.");
    return plan;
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Patient Dashboard</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Patients</h2>
          {patients.length > 0 ? (
            <select
              className="w-full border border-gray-300 rounded px-2 py-2 text-sm text-gray-900 bg-white"
              value={activePatientId || ''}
              onChange={(e) => setActivePatientId(e.target.value)}
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          ) : (
            <div className="text-sm text-gray-500">No patients</div>
          )}
        </section>

        <section className="bg-white border border-gray-200 rounded-lg p-4 lg:col-span-2">
          <h2 className="font-semibold text-gray-900 mb-2">Snapshot</h2>
          {demoPatient ? (
            <div className="text-sm text-gray-800 space-y-2">
              <div>Problems: {(demoPatient.problems || []).join(", ")}</div>
              <div>Medications: {(demoPatient.medications || []).join(", ")}</div>
              <div>Last Labs: {Object.entries(demoPatient.last_labs || {}).map(([k,v]) => `${k}: ${v}`).join(", ")}</div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">Loading patient snapshot…</div>
          )}
        </section>
      </div>

      {/* Patient Summary Panel */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Patient Summary</h2>
          <p className="text-xs text-gray-600 mb-2">Add any additional symptoms, changes, concerns, if any.</p>
          <textarea
            className="w-full border border-gray-300 rounded px-2 py-1 text-sm mb-2 text-gray-900 placeholder-gray-600"
            rows={3}
            placeholder="Add context here (optional)"
            value={summaryContext}
            onChange={(e) => setSummaryContext(e.target.value)}
          />
          <button
            onClick={async () => {
              setSummaryLoading(true);
              setSummary(null);
              try {
                // Build context from selected demo patient + optional user context
                const history: { sender: "user" | "assistant"; text: string }[] = [];
                const parts: string[] = [];
                if (demoPatient) {
                  const problems = (demoPatient.problems || []).join(', ');
                  const medications = (demoPatient.medications || []).join(', ');
                  const labs = Object.entries(demoPatient.last_labs || {}).map(([k, v]) => `${k}: ${v}`).join(', ');
                  if (problems) parts.push(`Problems: ${problems}`);
                  if (medications) parts.push(`Medications: ${medications}`);
                  if (labs) parts.push(`Last Labs: ${labs}`);
                }
                if (summaryContext.trim()) {
                  parts.push(`Context: ${summaryContext.trim()}`);
                }
                if (parts.length) {
                  history.push({ sender: "user", text: parts.join('\n') });
                }
                const conversation_history = history;
                const resp = await fetch(apiUrl("/v1/patient/summary"), {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ conversation_history, session_id: sessionId || undefined }),
                });
                const data = await resp.json();
                setSummary(data);
              } catch (_) {
                setSummary(null);
              } finally {
                setSummaryLoading(false);
              }
            }}
            className="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Generate Summary
          </button>

          {summaryLoading && <div className="text-xs text-gray-500 mt-2">Generating…</div>}
          {summary && (
            <div className="mt-3 text-sm text-gray-800 space-y-2">
              <div>
                <div className="font-semibold">Problems</div>
                <ul className="list-disc pl-5">
                  {(summary.problems || []).map((p: string, i: number) => (<li key={`pb-${i}`}>{p}</li>))}
                </ul>
              </div>
              <div>
                <div className="font-semibold">Medications</div>
                <ul className="list-disc pl-5">
                  {(summary.medications || []).map((m: string, i: number) => (<li key={`med-${i}`}>{m}</li>))}
                </ul>
              </div>
              <div>
                <div className="font-semibold">Labs</div>
                <div className="text-xs text-gray-700">Flags</div>
                <ul className="list-disc pl-5">
                  {((summary.labs && summary.labs.flags) || []).map((f: string, i: number) => (<li key={`lf-${i}`}>{f}</li>))}
                </ul>
                <div className="text-xs text-gray-700 mt-1">Notes</div>
                <ul className="list-disc pl-5">
                  {((summary.labs && summary.labs.notes) || []).map((n: string, i: number) => (<li key={`ln-${i}`}>{n}</li>))}
                </ul>
              </div>
              <div>
                <div className="font-semibold">Plan</div>
                <ul className="list-disc pl-5">
                  {(summary.plan || []).map((p: string, i: number) => (
                    <li key={`pl-${i}`}>{p}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </section>

        {/* Lab Interpretation Panel */}
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Lab Interpretation</h2>
          <div className="text-xs text-gray-600 mb-2">Upload lab PDFs/DOCX and get flagged insights.</div>
          <input
            id="patient-file-upload"
            type="file"
            multiple
            accept=".pdf,.docx"
            onChange={async (e) => {
              const files = e.target.files;
              if (!files || !files.length) return;
              setIsUploading(true);
              try {
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) formData.append('files', files[i]);
                const resp = await fetch(apiUrl('/upload-lab-results'), { method: 'POST', body: formData });
                const data = await resp.json();
                setSessionId(data.session_id);
                setUploadedFiles(Array.from(files));
              } catch (_) {
                // ignore
              } finally {
                setIsUploading(false);
              }
            }}
            className="hidden"
          />
          <label
            htmlFor="patient-file-upload"
            className={`mt-1 inline-block text-center py-2 px-3 rounded cursor-pointer transition-colors text-sm ${
              isUploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed pointer-events-none'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isUploading ? 'Uploading…' : 'Upload Lab Results'}
          </label>
          {isUploading && <div className="text-xs text-gray-500 mt-1">Uploading…</div>}
          {uploadedFiles.length > 0 && (
            <div className="mt-1 flex items-center justify-between">
              <div className="text-xs text-green-700">{uploadedFiles.length} file(s) uploaded</div>
              <button
                onClick={() => { setUploadedFiles([]); setSessionId(null); }}
                className="text-xs text-red-600 hover:bg-red-50 px-2 py-1 rounded"
                type="button"
              >
                Clear
              </button>
            </div>
          )}

          <div className="mt-3">
            <input
              type="text"
              className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-gray-900 placeholder-gray-600"
              placeholder="Optional question about these labs"
              value={labQuestion}
              onChange={(e) => setLabQuestion(e.target.value)}
            />
            <button
              onClick={async () => {
                if (!sessionId) return;
                setLabResult(null);
                const resp = await fetch(apiUrl('/v1/labs/interpret'), {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ session_id: sessionId, question: labQuestion || undefined })
                });
                const data = await resp.json();
                setLabResult(data);
              }}
              disabled={!sessionId}
              className="mt-2 px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              Interpret Labs
            </button>
          </div>

          {labResult && (
            <div className="mt-3 text-sm text-gray-800 space-y-2">
              <div>
                <div className="font-semibold">Flags</div>
                <ul className="list-disc pl-5">
                  {(labResult.flags || []).map((f: string, i: number) => (<li key={`rf-${i}`}>{f}</li>))}
                </ul>
              </div>
              <div>
                <div className="font-semibold">Explanations</div>
                <ul className="list-disc pl-5">
                  {(labResult.explanations || []).map((x: string, i: number) => (<li key={`rx-${i}`}>{x}</li>))}
                </ul>
              </div>
              <div>
                <div className="font-semibold">Recommendations</div>
                <ul className="list-disc pl-5">
                  {(labResult.recommendations || []).map((r: string, i: number) => (<li key={`rr-${i}`}>{r}</li>))}
                </ul>
              </div>
            </div>
          )}
        </section>
      </div>

      <div className="mt-6">
        <Link href="/" target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline">Chat with CareBridge Assistant</Link>
      </div>
    </div>
  );
}


