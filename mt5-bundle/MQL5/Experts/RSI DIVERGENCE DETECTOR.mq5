//+------------------------------------------------------------------+
//|                                    MTF_RSI_Divergence.mq5        |
//|         Multi-Timeframe RSI Divergence with Sync Dashboard       |
//|                                                                  |
//|  Sync Pairs:                                                     |
//|    Monthly -> Daily   |  Weekly -> H4   |  Daily -> H1           |
//|    H4 -> M15          |  H1 -> M5       |  M15 -> M1             |
//|                                                                  |
//|  Synced divergence = THICK TEAL (bull) / CRIMSON (bear)         |
//|  Solo divergence   = thin dashed white (bull) / red (bear)      |
//+------------------------------------------------------------------+
#property copyright   "MTF RSI Divergence"
#property version     "3.10"
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_label1  "RSI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrCrimson
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2
#property indicator_minimum 0
#property indicator_maximum 100

//=== INPUTS =============================================================
input group "=== RSI Settings ==="
input int                RSI_Period  = 14;
input ENUM_APPLIED_PRICE RSI_Price   = PRICE_CLOSE;

input group "=== Pivot Detection ==="
input int  PivotLookback  = 5;    // Bars each side for pivot confirmation (no repaint)
input int  DivLookback    = 150;  // Bars back to scan and keep on chart

input group "=== Divergence Filters ==="
input bool ShowRegularBull  = true;
input bool ShowRegularBear  = true;
input bool ShowHiddenBull   = true;
input bool ShowHiddenBear   = true;
input int  MinBarsBetween   = 3;
input double MinRSIDiff     = 2.0;  // Minimum RSI difference between pivots
input double MinPriceDiffPct = 0.05; // Minimum price move % between pivots

input group "=== Visual ==="
input int   DotSize          = 2;
input int   NormalLineWidth  = 1;
input int   SyncedLineWidth  = 3;
input color NormalBullCol    = clrWhite;
input color NormalBearCol    = clrTomato;
input color SyncedBullCol    = clrTeal;
input color SyncedBearCol    = clrCrimson;
input color PriceHighDotCol  = clrOrange;
input color PriceLowDotCol   = clrLimeGreen;
input color RSIHighDotCol    = clrOrangeRed;
input color RSILowDotCol     = clrDodgerBlue;

input group "=== Dashboard ==="
input bool  ShowDashboard  = true;
input int   DashX          = 20;
input int   DashY          = 30;

input group "=== Alerts ==="
input bool  AlertOnSync    = true;
input bool  AlertPopup     = false;
input bool  AlertMobile    = false;

//=======================================================================
// STRUCTS
//=======================================================================
struct DivInfo
{
   bool     detected;
   bool     isBull;
   bool     isHidden;
   double   strength;
   datetime timeA;
   datetime timeB;
   double   rsiA;
   double   rsiB;
};

struct SyncPair
{
   ENUM_TIMEFRAMES higherTF;
   ENUM_TIMEFRAMES lowerTF;
   string          label;
};

//=======================================================================
// GLOBALS
//=======================================================================
double   RSIBuffer[];
int      rsiHandle;
int      prevBars  = 0;
datetime lastAlert = 0;
string   PFX       = "MTFDIV_";

SyncPair syncPairs[6];

ENUM_TIMEFRAMES allTF[8];
int      rsiHandles[8];
DivInfo  tfDivBull[8];
DivInfo  tfDivBear[8];

// Dashboard colors (avoiding clrGold which is not standard MQL5)
color  DASH_BG       = C'15,15,25';
color  DASH_BORDER   = C'40,40,70';
color  DASH_TITLE    = clrWhite;
color  DASH_NEUTRAL  = C'160,160,180';
color  DASH_BULL     = clrLimeGreen;
color  DASH_BEAR     = clrTomato;
color  DASH_SYNC_B   = clrTeal;
color  DASH_SYNC_S   = clrCrimson;
color  DASH_GOLD     = C'218,165,32';   // Gold equivalent using RGB

//=======================================================================
int OnInit()
{
   SetIndexBuffer(0, RSIBuffer, INDICATOR_DATA);
   ArraySetAsSeries(RSIBuffer, true);

   rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, RSI_Price);
   if(rsiHandle == INVALID_HANDLE) { Alert("MTF RSI Div: RSI handle failed"); return INIT_FAILED; }

   IndicatorSetString (INDICATOR_SHORTNAME, "MTF RSI Div v3.1");
   IndicatorSetInteger(INDICATOR_DIGITS,    2);
   IndicatorSetInteger(INDICATOR_LEVELS,    2);
   IndicatorSetDouble (INDICATOR_LEVELVALUE,  0, 70.0);
   IndicatorSetDouble (INDICATOR_LEVELVALUE,  1, 30.0);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR,  0, clrDimGray);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR,  1, clrDimGray);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE,  0, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE,  1, STYLE_DOT);

   // Timeframe list
   allTF[0]=PERIOD_MN1; allTF[1]=PERIOD_W1;  allTF[2]=PERIOD_D1;  allTF[3]=PERIOD_H4;
   allTF[4]=PERIOD_H1;  allTF[5]=PERIOD_M15; allTF[6]=PERIOD_M5;  allTF[7]=PERIOD_M1;

   // Sync pairs
   syncPairs[0].higherTF=PERIOD_MN1; syncPairs[0].lowerTF=PERIOD_D1;  syncPairs[0].label="MN -> D1 ";
   syncPairs[1].higherTF=PERIOD_W1;  syncPairs[1].lowerTF=PERIOD_H4;  syncPairs[1].label="W1 -> H4 ";
   syncPairs[2].higherTF=PERIOD_D1;  syncPairs[2].lowerTF=PERIOD_H1;  syncPairs[2].label="D1 -> H1 ";
   syncPairs[3].higherTF=PERIOD_H4;  syncPairs[3].lowerTF=PERIOD_M15; syncPairs[3].label="H4 -> M15";
   syncPairs[4].higherTF=PERIOD_H1;  syncPairs[4].lowerTF=PERIOD_M5;  syncPairs[4].label="H1 -> M5 ";
   syncPairs[5].higherTF=PERIOD_M15; syncPairs[5].lowerTF=PERIOD_M1;  syncPairs[5].label="M15-> M1 ";

   // RSI handles for all 8 TFs
   for(int i=0; i<8; i++)
   {
      rsiHandles[i]         = iRSI(_Symbol, allTF[i], RSI_Period, RSI_Price);
      tfDivBull[i].detected = false;
      tfDivBear[i].detected = false;
   }

   return INIT_SUCCEEDED;
}

//=======================================================================
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, PFX);
   if(rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
   for(int i=0; i<8; i++)
      if(rsiHandles[i] != INVALID_HANDLE) IndicatorRelease(rsiHandles[i]);
}

//=======================================================================
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double   &open[],
                const double   &high[],
                const double   &low[],
                const double   &close[],
                const long     &tick_volume[],
                const long     &volume[],
                const int      &spread[])
{
   if(rates_total < RSI_Period + PivotLookback*2 + 10) return 0;
   if(CopyBuffer(rsiHandle, 0, 0, rates_total, RSIBuffer) <= 0) return 0;

   ArraySetAsSeries(RSIBuffer, true);
   ArraySetAsSeries(time,  true);
   ArraySetAsSeries(high,  true);
   ArraySetAsSeries(low,   true);

   if(rates_total == prevBars && prev_calculated > 0) return rates_total;
   prevBars = rates_total;

   ScanAllTimeframes();
   ObjectsDeleteAll(0, PFX);
   DrawCurrentTF(time, high, low, rates_total);
   if(ShowDashboard) DrawDashboard();

   return rates_total;
}

//=======================================================================
// HELPERS
//=======================================================================
int GetTFIndex(ENUM_TIMEFRAMES tf)
{
   for(int i=0; i<8; i++) if(allTF[i]==tf) return i;
   return -1;
}

string TFLabel(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_MN1: return "MN ";
      case PERIOD_W1:  return "W1 ";
      case PERIOD_D1:  return "D1 ";
      case PERIOD_H4:  return "H4 ";
      case PERIOD_H1:  return "H1 ";
      case PERIOD_M15: return "M15";
      case PERIOD_M5:  return "M5 ";
      case PERIOD_M1:  return "M1 ";
      default:         return "?  ";
   }
}

bool IsPivotHigh(const double &h[], int bar, int lb)
{
   int sz = ArraySize(h);
   if(bar-lb < 0 || bar+lb >= sz) return false;
   double v = h[bar];
   for(int i=1; i<=lb; i++) if(h[bar-i]>=v || h[bar+i]>=v) return false;
   return true;
}

bool IsPivotLow(const double &l[], int bar, int lb)
{
   int sz = ArraySize(l);
   if(bar-lb < 0 || bar+lb >= sz) return false;
   double v = l[bar];
   for(int i=1; i<=lb; i++) if(l[bar-i]<=v || l[bar+i]<=v) return false;
   return true;
}

//=======================================================================
// SCAN ALL 8 TIMEFRAMES
//=======================================================================
void ScanAllTimeframes()
{
   for(int t=0; t<8; t++)
   {
      tfDivBull[t].detected = false;
      tfDivBear[t].detected = false;

      int handle = rsiHandles[t];
      if(handle == INVALID_HANDLE) continue;

      int bars = Bars(_Symbol, allTF[t]);
      int needed = MathMin(DivLookback + PivotLookback*2 + 10, bars - RSI_Period - 2);
      if(needed < PivotLookback*2 + 5) continue;

      double rsiTF[], hTF[], lTF[];
      datetime tTF[];
      ArraySetAsSeries(rsiTF, true);
      ArraySetAsSeries(hTF,   true);
      ArraySetAsSeries(lTF,   true);
      ArraySetAsSeries(tTF,   true);

      if(CopyBuffer(rsiHandles[t], 0, 0, needed, rsiTF) <= 0) continue;
      if(CopyHigh(_Symbol, allTF[t], 0, needed, hTF)   <= 0) continue;
      if(CopyLow (_Symbol, allTF[t], 0, needed, lTF)   <= 0) continue;
      if(CopyTime(_Symbol, allTF[t], 0, needed, tTF)   <= 0) continue;

      int pivH[], pivL[];
      ArrayResize(pivH, 0);
      ArrayResize(pivL, 0);

      int scanEnd = MathMin(needed - PivotLookback - 1, ArraySize(hTF) - PivotLookback - 1);
      for(int i=PivotLookback; i<scanEnd; i++)
      {
         if(IsPivotHigh(hTF, i, PivotLookback)) { int s=ArraySize(pivH); ArrayResize(pivH,s+1); pivH[s]=i; }
         if(IsPivotLow (lTF, i, PivotLookback)) { int s=ArraySize(pivL); ArrayResize(pivL,s+1); pivL[s]=i; }
      }

      // Bear divergences
      int nh = ArraySize(pivH);
      for(int a=0; a<nh-1; a++)
      {
         int bA=pivH[a], bB=pivH[a+1];
         if(bB-bA < MinBarsBetween) continue;
         double pA=hTF[bA], pB=hTF[bB];
         double rA=rsiTF[bA], rB=rsiTF[bB];
         double rDiff = MathAbs(rA-rB);
         double pDiff = (pB > 0) ? MathAbs(pA-pB)/pB*100.0 : 0;
         if(rDiff < MinRSIDiff || pDiff < MinPriceDiffPct) continue;

         bool regBear = ShowRegularBear && pA > pB && rA < rB;
         bool hidBear = ShowHiddenBear  && pA < pB && rA > rB;
         if(regBear || hidBear)
         {
            tfDivBear[t].detected = true;
            tfDivBear[t].isBull   = false;
            tfDivBear[t].isHidden = hidBear;
            tfDivBear[t].strength = rDiff;
            tfDivBear[t].timeA    = tTF[bA];
            tfDivBear[t].timeB    = tTF[bB];
            tfDivBear[t].rsiA     = rA;
            tfDivBear[t].rsiB     = rB;
            break;
         }
      }

      // Bull divergences
      int nl = ArraySize(pivL);
      for(int a=0; a<nl-1; a++)
      {
         int bA=pivL[a], bB=pivL[a+1];
         if(bB-bA < MinBarsBetween) continue;
         double pA=lTF[bA], pB=lTF[bB];
         double rA=rsiTF[bA], rB=rsiTF[bB];
         double rDiff = MathAbs(rA-rB);
         double pDiff = (pB > 0) ? MathAbs(pA-pB)/pB*100.0 : 0;
         if(rDiff < MinRSIDiff || pDiff < MinPriceDiffPct) continue;

         bool regBull = ShowRegularBull && pA < pB && rA > rB;
         bool hidBull = ShowHiddenBull  && pA > pB && rA < rB;
         if(regBull || hidBull)
         {
            tfDivBull[t].detected = true;
            tfDivBull[t].isBull   = true;
            tfDivBull[t].isHidden = hidBull;
            tfDivBull[t].strength = rDiff;
            tfDivBull[t].timeA    = tTF[bA];
            tfDivBull[t].timeB    = tTF[bB];
            tfDivBull[t].rsiA     = rA;
            tfDivBull[t].rsiB     = rB;
            break;
         }
      }
   }
}

//=======================================================================
// DRAW CURRENT CHART TF
//=======================================================================
void DrawCurrentTF(const datetime &time[], const double &high[],
                   const double &low[], int total)
{
   ENUM_TIMEFRAMES curTF = Period();

   // Check if superior TF in any sync pair has divergence matching current TF
   bool syncedBull = false, syncedBear = false;
   for(int p=0; p<6; p++)
   {
      if(syncPairs[p].lowerTF == curTF)
      {
         int hIdx = GetTFIndex(syncPairs[p].higherTF);
         if(hIdx >= 0)
         {
            if(tfDivBull[hIdx].detected) syncedBull = true;
            if(tfDivBear[hIdx].detected) syncedBear = true;
         }
      }
   }

   int scanBars = MathMin(DivLookback + PivotLookback*2 + 5, total - RSI_Period - 2);

   int pivH[], pivL[];
   ArrayResize(pivH, 0);
   ArrayResize(pivL, 0);

   for(int i=PivotLookback; i<scanBars; i++)
   {
      if(IsPivotHigh(high, i, PivotLookback)) { int s=ArraySize(pivH); ArrayResize(pivH,s+1); pivH[s]=i; }
      if(IsPivotLow (low,  i, PivotLookback)) { int s=ArraySize(pivL); ArrayResize(pivL,s+1); pivL[s]=i; }
   }

   // Draw all pivot dots
   for(int k=0; k<ArraySize(pivH); k++) DrawDot(pivH[k], true,  time, high, low);
   for(int k=0; k<ArraySize(pivL); k++) DrawDot(pivL[k], false, time, high, low);

   // Draw ALL historical bear divergences
   int nh = ArraySize(pivH);
   for(int a=0; a<nh-1; a++)
   {
      int bA=pivH[a], bB=pivH[a+1];
      if(bB-bA < MinBarsBetween) continue;
      double pA=high[bA], pB=high[bB];
      double rA=RSIBuffer[bA], rB=RSIBuffer[bB];
      double rDiff=MathAbs(rA-rB);
      double pDiff=(pB>0) ? MathAbs(pA-pB)/pB*100.0 : 0;
      if(rDiff < MinRSIDiff || pDiff < MinPriceDiffPct) continue;

      bool regBear = ShowRegularBear && pA > pB && rA < rB;
      bool hidBear = ShowHiddenBear  && pA < pB && rA > rB;
      if(!regBear && !hidBear) continue;

      // Most recent bear on this TF: check sync
      bool isSynced = (a == 0) && syncedBear;
      color  lc = isSynced ? SyncedBearCol : NormalBearCol;
      int    lw = isSynced ? SyncedLineWidth : NormalLineWidth;
      string tag = "B" + IntegerToString(bA) + "x" + IntegerToString(bB);
      DrawDiv(bA, bB, true, hidBear, isSynced, lc, lw, time, high, low, tag, rDiff);

      if(a==0 && AlertOnSync && isSynced)
         DoAlert(false, hidBear, time[bA]);
   }

   // Draw ALL historical bull divergences
   int nl = ArraySize(pivL);
   for(int a=0; a<nl-1; a++)
   {
      int bA=pivL[a], bB=pivL[a+1];
      if(bB-bA < MinBarsBetween) continue;
      double pA=low[bA], pB=low[bB];
      double rA=RSIBuffer[bA], rB=RSIBuffer[bB];
      double rDiff=MathAbs(rA-rB);
      double pDiff=(pB>0) ? MathAbs(pA-pB)/pB*100.0 : 0;
      if(rDiff < MinRSIDiff || pDiff < MinPriceDiffPct) continue;

      bool regBull = ShowRegularBull && pA < pB && rA > rB;
      bool hidBull = ShowHiddenBull  && pA > pB && rA < rB;
      if(!regBull && !hidBull) continue;

      bool isSynced = (a == 0) && syncedBull;
      color  lc = isSynced ? SyncedBullCol : NormalBullCol;
      int    lw = isSynced ? SyncedLineWidth : NormalLineWidth;
      string tag = "U" + IntegerToString(bA) + "x" + IntegerToString(bB);
      DrawDiv(bA, bB, false, hidBull, isSynced, lc, lw, time, high, low, tag, rDiff);

      if(a==0 && AlertOnSync && isSynced)
         DoAlert(true, hidBull, time[bA]);
   }
}

//=======================================================================
// DRAW DOT
//=======================================================================
void DrawDot(int bar, bool isHigh, const datetime &time[],
             const double &high[], const double &low[])
{
   string pn = PFX + "PD" + (isHigh ? "H" : "L") + IntegerToString(bar);
   double pv = isHigh ? high[bar] : low[bar];
   color  pc = isHigh ? PriceHighDotCol : PriceLowDotCol;
   if(ObjectCreate(0, pn, OBJ_ARROW, 0, time[bar], pv))
   {
      ObjectSetInteger(0, pn, OBJPROP_ARROWCODE,  159);
      ObjectSetInteger(0, pn, OBJPROP_COLOR,      pc);
      ObjectSetInteger(0, pn, OBJPROP_WIDTH,      DotSize);
      ObjectSetInteger(0, pn, OBJPROP_ANCHOR,     isHigh ? ANCHOR_BOTTOM : ANCHOR_TOP);
      ObjectSetInteger(0, pn, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, pn, OBJPROP_HIDDEN,     true);
   }
   string rn = PFX + "RD" + (isHigh ? "H" : "L") + IntegerToString(bar);
   double rv = RSIBuffer[bar];
   color  rc = isHigh ? RSIHighDotCol : RSILowDotCol;
   if(ObjectCreate(0, rn, OBJ_ARROW, 1, time[bar], rv))
   {
      ObjectSetInteger(0, rn, OBJPROP_ARROWCODE,  159);
      ObjectSetInteger(0, rn, OBJPROP_COLOR,      rc);
      ObjectSetInteger(0, rn, OBJPROP_WIDTH,      DotSize);
      ObjectSetInteger(0, rn, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, rn, OBJPROP_HIDDEN,     true);
   }
}

//=======================================================================
// DRAW DIVERGENCE LINES + ARROWS
//=======================================================================
void DrawDiv(int bA, int bB, bool isBear, bool isHidden, bool isSynced,
             color lc, int lw, const datetime &time[],
             const double &high[], const double &low[],
             string tag, double strength)
{
   color  ac  = lc;
   int    aco = isBear ? 242 : 241;
   ENUM_LINE_STYLE lstyle = isSynced ? STYLE_SOLID : STYLE_DASH;

   double pA = isBear ? high[bA] : low[bA];
   double pB = isBear ? high[bB] : low[bB];
   double rA = RSIBuffer[bA];
   double rB = RSIBuffer[bB];

   // Price line
   string lp = PFX + "LP" + tag;
   if(ObjectCreate(0, lp, OBJ_TREND, 0, time[bB], pB, time[bA], pA))
   {
      ObjectSetInteger(0, lp, OBJPROP_COLOR,      lc);
      ObjectSetInteger(0, lp, OBJPROP_WIDTH,      lw);
      ObjectSetInteger(0, lp, OBJPROP_STYLE,      lstyle);
      ObjectSetInteger(0, lp, OBJPROP_RAY_RIGHT,  false);
      ObjectSetInteger(0, lp, OBJPROP_RAY_LEFT,   false);
      ObjectSetInteger(0, lp, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lp, OBJPROP_HIDDEN,     true);
   }

   // RSI line
   string lr = PFX + "LR" + tag;
   if(ObjectCreate(0, lr, OBJ_TREND, 1, time[bB], rB, time[bA], rA))
   {
      ObjectSetInteger(0, lr, OBJPROP_COLOR,      lc);
      ObjectSetInteger(0, lr, OBJPROP_WIDTH,      lw);
      ObjectSetInteger(0, lr, OBJPROP_STYLE,      lstyle);
      ObjectSetInteger(0, lr, OBJPROP_RAY_RIGHT,  false);
      ObjectSetInteger(0, lr, OBJPROP_RAY_LEFT,   false);
      ObjectSetInteger(0, lr, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lr, OBJPROP_HIDDEN,     true);
   }

   // Price arrow
   string ap = PFX + "AP" + tag;
   if(ObjectCreate(0, ap, OBJ_ARROW, 0, time[bA], pA))
   {
      ObjectSetInteger(0, ap, OBJPROP_ARROWCODE,  aco);
      ObjectSetInteger(0, ap, OBJPROP_COLOR,      ac);
      ObjectSetInteger(0, ap, OBJPROP_WIDTH,      isSynced ? 4 : 2);
      ObjectSetInteger(0, ap, OBJPROP_ANCHOR,     isBear ? ANCHOR_BOTTOM : ANCHOR_TOP);
      ObjectSetInteger(0, ap, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, ap, OBJPROP_HIDDEN,     true);
   }

   // RSI arrow
   string ar = PFX + "AR" + tag;
   if(ObjectCreate(0, ar, OBJ_ARROW, 1, time[bA], rA))
   {
      ObjectSetInteger(0, ar, OBJPROP_ARROWCODE,  aco);
      ObjectSetInteger(0, ar, OBJPROP_COLOR,      ac);
      ObjectSetInteger(0, ar, OBJPROP_WIDTH,      isSynced ? 4 : 2);
      ObjectSetInteger(0, ar, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, ar, OBJPROP_HIDDEN,     true);
   }

   // Strength label
   string lb = PFX + "LB" + tag;
   string lblTxt = (isSynced ? "* " : "") + (isHidden ? "H " : "R ") + DoubleToString(strength, 1);
   if(ObjectCreate(0, lb, OBJ_TEXT, 0, time[bA], pA))
   {
      ObjectSetString (0, lb, OBJPROP_TEXT,       lblTxt);
      ObjectSetInteger(0, lb, OBJPROP_COLOR,      ac);
      ObjectSetInteger(0, lb, OBJPROP_FONTSIZE,   isSynced ? 8 : 7);
      ObjectSetInteger(0, lb, OBJPROP_ANCHOR,     isBear ? ANCHOR_LEFT_UPPER : ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0, lb, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lb, OBJPROP_HIDDEN,     true);
   }
}

//=======================================================================
// DASHBOARD
//=======================================================================
void MakeRect(string name, int x, int y, int w, int h, color bg, color border)
{
   string n = PFX + "BG_" + name;
   if(ObjectCreate(0, n, OBJ_RECTANGLE_LABEL, 0, 0, 0))
   {
      ObjectSetInteger(0, n, OBJPROP_XDISTANCE,   x);
      ObjectSetInteger(0, n, OBJPROP_YDISTANCE,   y);
      ObjectSetInteger(0, n, OBJPROP_XSIZE,       w);
      ObjectSetInteger(0, n, OBJPROP_YSIZE,       h);
      ObjectSetInteger(0, n, OBJPROP_BGCOLOR,     bg);
      ObjectSetInteger(0, n, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, n, OBJPROP_COLOR,       border);
      ObjectSetInteger(0, n, OBJPROP_WIDTH,       1);
      ObjectSetInteger(0, n, OBJPROP_BACK,        true);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE,  false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN,      true);
   }
}

void MakeLabel(string name, int x, int y, string txt, color clr, int sz=8)
{
   string n = PFX + "LB_" + name;
   if(ObjectCreate(0, n, OBJ_LABEL, 0, 0, 0))
   {
      ObjectSetInteger(0, n, OBJPROP_XDISTANCE,  x);
      ObjectSetInteger(0, n, OBJPROP_YDISTANCE,  y);
      ObjectSetString (0, n, OBJPROP_TEXT,        txt);
      ObjectSetInteger(0, n, OBJPROP_COLOR,       clr);
      ObjectSetInteger(0, n, OBJPROP_FONTSIZE,    sz);
      ObjectSetString (0, n, OBJPROP_FONT,        "Consolas");
      ObjectSetInteger(0, n, OBJPROP_CORNER,      CORNER_LEFT_UPPER);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE,  false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN,      true);
   }
}

string StrBar(double s)
{
   int lvl = (int)MathMin(5, MathMax(1, MathRound(s / 5.0)));
   string b = "";
   for(int i=0; i<lvl;  i++) b += "|";
   for(int i=lvl; i<5;  i++) b += ".";
   return b;
}

void DrawDashboard()
{
   int x  = DashX;
   int y  = DashY;
   int rh = 15;
   int w  = 300;
   int totalRows = 12 + 6 + 10; // header + pairs + TF RSI rows + padding
   int h  = totalRows * rh + 10;

   MakeRect("MAIN", x, y, w, h, DASH_BG, DASH_BORDER);

   int cx = x + 8;
   int cy = y + 6;

   // Title
   MakeLabel("TITLE", cx, cy, "  MTF RSI DIVERGENCE  v3.1", DASH_TITLE, 9);
   cy += rh + 2;
   MakeLabel("SEP1",  cx, cy, "-------------------------------", DASH_BORDER, 7);
   cy += rh - 3;

   // Column header
   MakeLabel("CH1", cx,       cy, "PAIR     ", DASH_NEUTRAL, 8);
   MakeLabel("CH2", cx + 85,  cy, "BULL",      DASH_BULL,    8);
   MakeLabel("CH3", cx + 155, cy, "BEAR",      DASH_BEAR,    8);
   MakeLabel("CH4", cx + 225, cy, "SYNC",      DASH_GOLD,    8);
   cy += rh;

   ENUM_TIMEFRAMES curTF = Period();

   for(int p=0; p<6; p++)
   {
      int hIdx = GetTFIndex(syncPairs[p].higherTF);
      int lIdx = GetTFIndex(syncPairs[p].lowerTF);

      bool hBull = (hIdx>=0) && tfDivBull[hIdx].detected;
      bool hBear = (hIdx>=0) && tfDivBear[hIdx].detected;
      bool lBull = (lIdx>=0) && tfDivBull[lIdx].detected;
      bool lBear = (lIdx>=0) && tfDivBear[lIdx].detected;
      bool syncBull = hBull && lBull;
      bool syncBear = hBear && lBear;
      bool isCurPair = (syncPairs[p].lowerTF==curTF || syncPairs[p].higherTF==curTF);

      color rowCol = isCurPair ? clrWhite : DASH_NEUTRAL;
      string pid = IntegerToString(p);

      MakeLabel("PR"+pid, cx, cy, syncPairs[p].label, rowCol, 8);

      string bullTxt = hBull ? StrBar(tfDivBull[hIdx].strength)+(tfDivBull[hIdx].isHidden?"H":"R") : "-----";
      color  bullCol = syncBull ? DASH_SYNC_B : (hBull ? DASH_BULL : DASH_NEUTRAL);
      MakeLabel("PB"+pid, cx+85, cy, bullTxt, bullCol, 8);

      string bearTxt = hBear ? StrBar(tfDivBear[hIdx].strength)+(tfDivBear[hIdx].isHidden?"H":"R") : "-----";
      color  bearCol = syncBear ? DASH_SYNC_S : (hBear ? DASH_BEAR : DASH_NEUTRAL);
      MakeLabel("PS"+pid, cx+155, cy, bearTxt, bearCol, 8);

      string syncTxt = "  -";
      color  syncCol = DASH_NEUTRAL;
      if(syncBull) { syncTxt = "* BULL"; syncCol = DASH_SYNC_B; }
      if(syncBear) { syncTxt = "* BEAR"; syncCol = DASH_SYNC_S; }
      if(syncBull && syncBear) { syncTxt = "* BOTH"; syncCol = DASH_GOLD; }
      MakeLabel("PY"+pid, cx+225, cy, syncTxt, syncCol, 8);

      cy += rh - 1;
   }

   cy += 2;
   MakeLabel("SEP2", cx, cy, "-------------------------------", DASH_BORDER, 7);
   cy += rh - 3;

   // RSI per TF header
   MakeLabel("RH", cx, cy, "TF    RSI    DIV STATE", DASH_NEUTRAL, 8);
   cy += rh;

   for(int t=0; t<8; t++)
   {
      double rArr[];
      ArraySetAsSeries(rArr, true);
      double curRSI = 50.0;
      if(rsiHandles[t] != INVALID_HANDLE)
         if(CopyBuffer(rsiHandles[t], 0, 0, 3, rArr) > 0) curRSI = rArr[0];

      bool isCurTF = (allTF[t] == curTF);
      color tfCol  = isCurTF ? clrWhite : DASH_NEUTRAL;
      color rsiCol = (curRSI > 70) ? DASH_BEAR : ((curRSI < 30) ? DASH_BULL : tfCol);

      string divTxt = "  none";
      color  divCol = DASH_NEUTRAL;
      if(tfDivBull[t].detected && tfDivBear[t].detected) { divTxt = "^ v BOTH"; divCol = DASH_GOLD; }
      else if(tfDivBull[t].detected) { divTxt = "^  BULL"; divCol = DASH_BULL; }
      else if(tfDivBear[t].detected) { divTxt = "v  BEAR"; divCol = DASH_BEAR; }

      string tid = IntegerToString(t);
      MakeLabel("TL"+tid, cx,      cy, TFLabel(allTF[t]),               tfCol,  8);
      MakeLabel("TV"+tid, cx+35,   cy, StringFormat("%-6.1f", curRSI),  rsiCol, 8);
      MakeLabel("TD"+tid, cx+100,  cy, divTxt,                          divCol, 8);
      cy += rh - 1;
   }

   // Footer
   cy += 3;
   string footer = "  TF: " + TFLabel(curTF) + "  RSI(" + IntegerToString(RSI_Period) + ")  Pivot:" + IntegerToString(PivotLookback);
   MakeLabel("FOOT", cx, cy, footer, DASH_NEUTRAL, 7);
}

//=======================================================================
// ALERTS
//=======================================================================
void DoAlert(bool isBull, bool isHidden, datetime barTime)
{
   if(!AlertPopup && !AlertMobile) return;
   if(barTime == lastAlert) return;
   lastAlert = barTime;
   string kind = (isHidden ? "Hidden " : "Regular ") + (isBull ? "Bullish" : "Bearish");
   string msg  = _Symbol + " " + EnumToString(Period()) + " | SYNCED RSI Div: " + kind;
   if(AlertPopup)  Alert(msg);
   if(AlertMobile) SendNotification(msg);
}
//+------------------------------------------------------------------+