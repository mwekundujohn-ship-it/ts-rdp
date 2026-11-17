//+------------------------------------------------------------------+
//|  ChartReaderEA.mq5                                               |
//|  Live chart -> JSON bridge for opencode.                         |
//|  Writes MQL5\Files\opencode\chart_state.json every InpRefreshSec |
//|  seconds: symbol/quote/account + every indicator buffer value.   |
//|                                                                  |
//|  LIVE USE ONLY. Do NOT run in the Strategy Tester: the 1-second  |
//|  timer fires once per SIMULATED second and will flood the disk.  |
//+------------------------------------------------------------------+
#property copyright "opencode"
#property version   "1.00"
#property strict

input int    InpRefreshSec = 1;     // refresh interval (seconds)
input int    InpBars       = 5;     // values per buffer
input int    InpMaxBuffers = 8;     // max buffers per indicator

string g_last = "";

//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(InpRefreshSec > 0 ? InpRefreshSec : 1);
   WriteState();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   WriteState();
}

//+------------------------------------------------------------------+
string JsonStr(const string s)
{
   string o = s;
   StringReplace(o, "\\", "\\\\");
   StringReplace(o, "\"", "\\\"");
   StringReplace(o, "\n", " ");
   StringReplace(o, "\r", " ");
   return("\"" + o + "\"");
}

string TfName(ENUM_TIMEFRAMES tf)
{
   return(EnumToString(tf));
}

string BuildJson()
{
   string j = "{";
   j += "\"source\":\"ChartReaderEA\"";
   j += ",\"ts\":" + (string)(long)TimeCurrent();
   j += ",\"gmt_ts\":" + (string)(long)TimeGMT();
   j += ",\"symbol\":" + JsonStr(_Symbol);
   j += ",\"timeframe\":" + JsonStr(TfName((ENUM_TIMEFRAMES)Period()));
   j += ",\"period_seconds\":" + (string)PeriodSeconds();
   j += ",\"digits\":" + (string)_Digits;
   j += ",\"point\":" + DoubleToString(_Point, 8);
   MqlTick tk;
   double bid = 0, ask = 0;
   if(SymbolInfoTick(_Symbol, tk)) { bid = tk.bid; ask = tk.ask; }
   j += ",\"bid\":" + DoubleToString(bid, _Digits);
   j += ",\"ask\":" + DoubleToString(ask, _Digits);
   j += ",\"spread\":" + (string)(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   j += ",\"bars\":" + (string)Bars(_Symbol, PERIOD_CURRENT);

   j += ",\"account\":{";
   j += "\"login\":" + (string)AccountInfoInteger(ACCOUNT_LOGIN);
   j += ",\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2);
   j += ",\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2);
   j += ",\"currency\":" + JsonStr(AccountInfoString(ACCOUNT_CURRENCY));
   j += ",\"leverage\":" + (string)AccountInfoInteger(ACCOUNT_LEVERAGE);
   j += "}";

   j += ",\"windows\":[";
   int wins = (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL);
   for(int w = 0; w < wins; w++)
   {
      if(w > 0) j += ",";
      j += "{\"index\":" + (string)w;
      j += ",\"subwindow\":" + (string)w;
      j += ",\"indicators\":[";
      int total = ChartIndicatorsTotal(0, w);
      for(int i = 0; i < total; i++)
      {
         if(i > 0) j += ",";
         string name = ChartIndicatorName(0, w, i);
         int handle = ChartIndicatorGet(0, w, i);
         j += "{\"name\":" + JsonStr(name);
         j += ",\"subwindow\":" + (string)w;
         j += ",\"handle\":" + (string)handle;
         j += ",\"buffers\":[";
         int nb = (handle != INVALID_HANDLE) ? BarsCalculated(handle) : 0;
         int nbuf = (int)indicator_buffers_num(handle);
         if(nbuf <= 0) nbuf = 1;
         if(nbuf > InpMaxBuffers) nbuf = InpMaxBuffers;
         for(int b = 0; b < nbuf; b++)
         {
            if(b > 0) j += ",";
            double buf[];
            int copied = CopyBuffer(handle, b, 0, InpBars, buf);
            j += "{\"index\":" + (string)b + ",\"values\":[";
            for(int k = 0; k < copied; k++)
            {
               if(k > 0) j += ",";
               if(buf[k] == EMPTY_VALUE) j += "null";
               else j += DoubleToString(buf[k], _Digits);
            }
            j += "]}";
         }
         j += "]";
         j += "}";
      }
      j += "]";
      j += "}";
   }
   j += "]";
   j += "}";
   return(j);
}

int indicator_buffers_num(int handle)
{
   if(handle == INVALID_HANDLE) return(0);
   int n = 0;
   double tmp[];
   for(int b = 0; b < 32; b++)
   {
      int got = CopyBuffer(handle, b, 0, 1, tmp);
      if(got < 0 && GetLastError() == ERR_INDICATOR_DATA_NOT_FOUND)
         break;
      n++;
   }
   return(n);
}

void WriteState()
{
   string j = BuildJson();
   if(j == g_last) return;
   g_last = j;
   int h = FileOpen("opencode\\chart_state.json", FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(h == INVALID_HANDLE) return;
   FileWriteString(h, j);
   FileFlush(h);
   FileClose(h);
}
//+------------------------------------------------------------------+
