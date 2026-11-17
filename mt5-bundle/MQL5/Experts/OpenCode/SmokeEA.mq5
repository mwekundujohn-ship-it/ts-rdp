//+------------------------------------------------------------------+
//|  SmokeEA.mq5                                                     |
//|  Minimal EMA(10/30) cross test expert. One position at a time.   |
//|  Purpose: prove the headless compile + backtest pipeline works.  |
//+------------------------------------------------------------------+
#property copyright "opencode"
#property version   "1.00"
#property strict

input double InpLots    = 0.10;  // volume
input int    InpFast    = 10;    // fast EMA
input int    InpSlow    = 30;    // slow EMA
input int    InpMagic   = 99001; // magic

int    g_fast = INVALID_HANDLE;
int    g_slow = INVALID_HANDLE;
datetime g_lastBar = 0;

int OnInit()
{
   g_fast = iMA(_Symbol, PERIOD_CURRENT, InpFast, 0, MODE_EMA, PRICE_CLOSE);
   g_slow = iMA(_Symbol, PERIOD_CURRENT, InpSlow, 0, MODE_EMA, PRICE_CLOSE);
   if(g_fast == INVALID_HANDLE || g_slow == INVALID_HANDLE)
      return(INIT_FAILED);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_fast != INVALID_HANDLE) IndicatorRelease(g_fast);
   if(g_slow != INVALID_HANDLE) IndicatorRelease(g_slow);
}

double StopLevelPrice()
{
   long stoplevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   return((double)stoplevel * _Point);
}

void OnTick()
{
   datetime t = (datetime)SeriesInfoInteger(_Symbol, PERIOD_CURRENT, SERIES_LASTBAR_DATE);
   if(t == g_lastBar) return;
   g_lastBar = t;

   double fast[], slow[];
   if(CopyBuffer(g_fast, 0, 1, 2, fast) < 2) return;
   if(CopyBuffer(g_slow, 0, 1, 2, slow) < 2) return;

   bool crossUp   = (fast[0] > slow[0] && fast[1] <= slow[1]) || (fast[1] > slow[1] && fast[0] > slow[0]);
   bool crossDown = (fast[0] < slow[0] && fast[1] >= slow[1]) || (fast[1] < slow[1] && fast[0] < slow[0]);

   bool hasPos = PositionSelect(_Symbol);
   if(hasPos && PositionGetInteger(POSITION_MAGIC) != InpMagic) hasPos = false;

   if(!hasPos && crossUp)   Open(ORDER_TYPE_BUY);
   if(!hasPos && crossDown) Open(ORDER_TYPE_SELL);
}

void Open(ENUM_ORDER_TYPE type)
{
   MqlTradeRequest req; MqlTradeResult res;
   ZeroMemory(req); ZeroMemory(res);
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                           : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   req.action   = TRADE_ACTION_DEAL;
   req.symbol   = _Symbol;
   req.volume   = InpLots;
   req.type     = type;
   req.price    = price;
   req.deviation= 20;
   req.magic    = InpMagic;
   req.type_filling = Filling();
   OrderSend(req, res);
}

ENUM_ORDER_TYPE_FILLING Filling()
{
   long mode = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((mode & SYMBOL_FILLING_FOK) != 0) return(ORDER_FILLING_FOK);
   if((mode & SYMBOL_FILLING_IOC) != 0) return(ORDER_FILLING_IOC);
   return(ORDER_FILLING_RETURN);
}
//+------------------------------------------------------------------+
