"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/api";
import Link from "next/link";

interface Msg { sender: "user" | "assistant"; text: string }

export default function ProviderPage() {
  const [patients, setPatients] = useState<Record<string, any>[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [detail, setDetail] = useState<Record<string, any> | null>(null);
  const [note, setNote] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);
  const [template, setTemplate] = useState<string>("general");

  useEffect(() => {
    fetch(apiUrl("/v1/demo/patients"))
      .then((r) => r.json())
      .then((list) => {
        setPatients(list);
        if (list.length) setSelected(list[0].id);
      });
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetch(apiUrl(`/v1/demo/patients/${selected}`))
      .then((r) => r.json())
      .then(setDetail);
  }, [selected]);

  const generateNote = async () => {
    const messages: Msg[] = [
      { sender: "user", text: `Template: ${template}. Summarize patient ${selected} visit and create HPI/Assessment/Plan.` },
    ];
    const resp = await fetch(apiUrl("/v1/provider/note"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    const data = await resp.json();
    setNote(data.markdown || "");
  };

  const downloadNote = () => {
    if (!note) return;
    const blob = new Blob([note], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `carebridge-note-${selected || 'patient'}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const copyNote = async () => {
    if (!note) return;
    try {
      await navigator.clipboard.writeText(note);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (_) {}
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Provider Workspace</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Patients</h2>
          <select
            className="w-full border border-gray-300 rounded px-2 py-1 text-sm text-gray-900 bg-white"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </section>

        <section className="bg-white border border-gray-200 rounded-lg p-4 lg:col-span-2">
          <h2 className="font-semibold text-gray-900 mb-2">Patient Summary</h2>
          {detail ? (
            <div className="text-sm text-gray-800 space-y-2">
              <div>Problems: {(detail.problems || []).join(", ")}</div>
              <div>Medications: {(detail.medications || []).join(", ")}</div>
              <div>Last Labs: {Object.entries(detail.last_labs || {}).map(([k,v]) => `${k}: ${v}`).join(", ")}</div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">Select a patient…</div>
          )}
        </section>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Generate Note</h2>
          <div className="flex items-center gap-2">
            <select
              className="border border-gray-300 rounded px-2 py-1 text-sm text-gray-900 bg-white"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              title="Note template"
            >
              <option value="general">General visit</option>
              <option value="followup">Follow-up visit</option>
              <option value="discharge">Discharge summary</option>
            </select>
            <button onClick={generateNote} className="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">Generate HPI/Assessment/Plan</button>
            <button onClick={copyNote} disabled={!note} className="px-3 py-2 text-sm bg-gray-100 text-gray-800 rounded hover:bg-gray-200 disabled:opacity-50">{copied ? 'Copied!' : 'Copy'}</button>
            <button onClick={downloadNote} disabled={!note} className="px-3 py-2 text-sm bg-gray-100 text-gray-800 rounded hover:bg-gray-200 disabled:opacity-50">Download .md</button>
          </div>
          <pre className="mt-3 text-xs whitespace-pre-wrap text-gray-800">{note}</pre>
        </section>
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold text-gray-900 mb-2">Open Chat</h2>
          <Link href="/" target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline">Chat with CareBridge Assistant</Link>
        </section>
      </div>
    </div>
  );
}


