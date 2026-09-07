import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

function App() {
  const [stats, setStats] = useState(null);
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("ivf");
  const [k, setK] = useState(10);
  const [nprobe, setNprobe] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/stats`)
      .then((res) => res.json())
      .then((data) => setStats(data))
      .catch(() => setError("Backend is not running"));
  }, []);

  const search = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: query,
          k: Number(k),
          method,
          nprobe: Number(nprobe),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Search failed");
      }

      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090d14] text-zinc-100 font-sans selection:bg-emerald-500 selection:text-black">
      
      {/* Header */}
      <header className="border-b border-zinc-800/80 bg-[#0c111c]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center font-bold text-emerald-400">
              V
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                NeuralVec Studio
                <span className="text-[10px] bg-emerald-950/80 text-emerald-400 border border-emerald-800/50 px-2 py-0.5 rounded font-mono">
                  v2.0
                </span>
              </h1>
              <p className="text-xs text-zinc-400">
                In-Memory Vector Retrieval System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-full">
            <span className={`h-2 w-2 rounded-full ${error ? "bg-rose-500 animate-pulse" : "bg-emerald-400"}`}></span>
            <span className="text-xs text-zinc-300 font-mono">
              {error ? "API Disconnected" : "API Active"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">

        {/* Stats */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          
          <StatCard
            title="Vectors"
            value={stats?.active_vectors ?? "..."}
            subtitle="Active vectors in memory"
          />

          <StatCard
            title="Dimensions"
            value={stats?.dimension ?? "..."}
            subtitle="Vector dimensions"
          />

          <StatCard
            title="Clusters"
            value={stats?.ivf_clusters ?? "..."}
            subtitle="IVF partitioning"
          />

          <StatCard
            title="Index Mode"
            value="IVF-Flat"
            subtitle="Approximate search"
          />

        </div>

        {/* Search Input Section */}
        <section className="mt-8 rounded-2xl border border-zinc-800/80 bg-[#0c111c] p-6 shadow-xl">
          
          <div className="mb-6 flex justify-between items-start">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Vector Search
              </h2>
              <p className="mt-1 text-xs text-zinc-400">
                Search documents using semantic similarity.
              </p>
            </div>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/50 border border-emerald-800/40 px-2.5 py-1 rounded-md">
              384D Space
            </span>
          </div>

          <div className="flex flex-col gap-4">
            
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your search query here"
              rows={3}
              className="w-full resize-none rounded-xl border border-zinc-800 bg-[#06090e] px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/60 font-mono"
            />

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              
              <div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-400 font-mono">
                  Search Method
                </label>

                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full rounded-xl border border-zinc-800 bg-[#06090e] px-3 py-2 text-xs text-zinc-200 outline-none focus:border-emerald-500"
                >
                  <option value="ivf">IVF-Flat (Approximate)</option>
                  <option value="brute">Brute Force (Exact Ground Truth)</option>
                </select>
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-400 font-mono">
                  Top K
                </label>

                <input
                  type="number"
                  min="1"
                  max="100"
                  value={k}
                  onChange={(e) => setK(e.target.value)}
                  className="w-full rounded-xl border border-zinc-800 bg-[#06090e] px-3 py-2 text-xs text-zinc-200 outline-none focus:border-emerald-500 font-mono"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-400 font-mono">
                  Nprobe
                </label>

                <input
                  type="number"
                  min="1"
                  value={nprobe}
                  onChange={(e) => setNprobe(e.target.value)}
                  disabled={method === "brute"}
                  className="w-full rounded-xl border border-zinc-800 bg-[#06090e] px-3 py-2 text-xs text-zinc-200 outline-none focus:border-emerald-500 disabled:opacity-30 font-mono"
                />
              </div>

            </div>

            <div className="flex justify-end mt-2">
              <button
                onClick={search}
                disabled={loading || !query.trim()}
                className="rounded-xl bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-bold text-xs px-6 py-3 transition shadow-lg shadow-emerald-500/10 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Computing Nearest Neighbors..." : "Execute Search ↵"}
              </button>
            </div>

          </div>

          {error && (
            <div className="mt-4 rounded-xl border border-rose-900/50 bg-rose-950/20 px-4 py-3 text-xs text-rose-300 font-mono">
              ⚠️ {error}
            </div>
          )}
        </section>

        {/* Results */}
        <section className="mt-8">
          
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wider">
                Matches Returned ({results.length})
              </h2>
            </div>
          </div>

          {results.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-zinc-800 bg-[#0c111c]/50 p-12 text-center">
              <p className="text-xs text-zinc-500 font-mono">
                No active search results. Enter a query prompt above.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((result, index) => (
                <div
                  key={result.id}
                  className="rounded-xl border border-zinc-800/80 bg-[#0c111c] p-4 transition hover:border-zinc-700/80 flex items-start justify-between gap-4 group"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs font-mono font-semibold text-emerald-400">
                      #{index + 1}
                    </div>

                    <div>
                      <p className="text-sm text-zinc-200 group-hover:text-emerald-100 transition leading-relaxed">
                        {result.text || "No document text available"}
                      </p>

                      <p className="mt-2 text-[10px] text-zinc-500 font-mono">
                        Vector ID: {result.id}
                      </p>
                    </div>
                  </div>

                  <div className="shrink-0 text-right font-mono">
                    <p className="text-sm font-bold text-emerald-400 bg-emerald-950/50 border border-emerald-800/40 px-2.5 py-1 rounded-md">
                      {typeof result.score === 'number' ? result.score.toFixed(4) : result.score}
                    </p>
                    <p className="text-[10px] text-zinc-500 mt-1">
                      Cosine Sim
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}

        </section>

        {/* Footer */}
        <footer className="mt-12 border-t border-zinc-800/60 py-6 text-center text-xs font-mono text-zinc-600">
          NeuralVec Core · Brute-Force & IVF-Flat Indexing
        </footer>

      </main>
    </div>
  );
}

function StatCard({ title, value, subtitle }) {
  return (
    <div className="rounded-2xl border border-zinc-800/80 bg-[#0c111c] p-5 shadow-lg">
      <p className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
        {title}
      </p>

      <p className="mt-2 text-2xl font-bold text-emerald-400 font-mono">
        {value}
      </p>

      <p className="mt-1 text-[11px] text-zinc-500">
        {subtitle}
      </p>
    </div>
  );
}

export default App;