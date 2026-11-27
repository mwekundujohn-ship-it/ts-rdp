// Auto-loaded by opencode. Tools: mt5_eas, mt5_chart, mt5_compile, mt5_backtest
import { spawn } from "node:child_process";
import * as path from "node:path";
import * as os from "node:os";

const HOME = process.env.MT5_OPENCODE_HOME || "C:\\mt5-opencode";
const PY = process.env.MT5_PYTHON || "python";
const SCRIPTS = path.join(HOME, "scripts");
const MT5_PY = path.join(SCRIPTS, "mt5.py");
const READ_PY = path.join(SCRIPTS, "read_chart.py");

function runPy(pyfile: string, args: string[], timeoutMs=180000): Promise<{ code: number|null, stdout: string, stderr: string }> {
  return new Promise((resolve) => {
    const p = spawn(PY, [pyfile, ...args], { cwd: HOME, windowsHide: true });
    let stdout="", stderr="";
    const t = setTimeout(() => { try{p.kill();}catch{} resolve({code:null,stdout,stderr:"TIMEOUT\n"+stderr}); }, timeoutMs);
    p.stdout.on("data", d=>stdout+=d.toString());
    p.stderr.on("data", d=>stderr+=d.toString());
    p.on("close", code=>{ clearTimeout(t); resolve({code,stdout,stderr}); });
  });
}

export const mt5_eas = async () => {
  const r = await runPy(MT5_PY, ["eas"], 60000);
  return { tool: "mt5_eas", ...r };
};
export const mt5_chart = async (args?: { json?: boolean, pretty?: boolean }) => {
  const a:string[]=[];
  if(args?.json) a.push("--json");
  if(args?.pretty) a.push("--pretty");
  const r = await runPy(READ_PY, a, 30000);
  return { tool: "mt5_chart", ...r };
};
export const mt5_compile = async (args: { source: string }) => {
  if(!args?.source) throw new Error("source required");
  const r = await runPy(MT5_PY, ["compile", args.source], 120000);
  return { tool: "mt5_compile", ...r };
};
export const mt5_backtest = async (args: { ea: string, symbol: string, period: string, from: string, to: string, model?: string, deposit?: number, leverage?: number, inputs?: string[] }) => {
  const a:string[]=["backtest","--ea",args.ea,"--symbol",args.symbol,"--period",args.period,"--from",args.from,"--to",args.to];
  if(args.model) a.push("--model",args.model);
  if(args.deposit!==undefined) a.push("--deposit",String(args.deposit));
  if(args.leverage!==undefined) a.push("--leverage",String(args.leverage));
  if(args.inputs) for(const x of args.inputs) a.push("--input",x);
  const r = await runPy(MT5_PY, a, 600000);
  return { tool: "mt5_backtest", ...r };
};
export const mt5 = mt5_eas;
