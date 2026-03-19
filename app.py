from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests as req
import time
import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY    = os.getenv('ZERODHA_API_KEY')
API_SECRET = os.getenv('ZERODHA_API_SECRET')
print("API_KEY:", API_KEY)
print("API_SECRET:", API_SECRET)
kite_sessions = {}  # stores access tokens per user
app = Flask(__name__)
CORS(app)

STOCKS = {
    'RELIANCE':'RELIANCE.NS','HDFCBANK':'HDFCBANK.NS','TCS':'TCS.NS',
    'INFY':'INFY.NS','ICICIBANK':'ICICIBANK.NS','HINDUNILVR':'HINDUNILVR.NS',
    'ITC':'ITC.NS','SBIN':'SBIN.NS','BHARTIARTL':'BHARTIARTL.NS',
    'KOTAKBANK':'KOTAKBANK.NS','BAJFINANCE':'BAJFINANCE.NS','ASIANPAINT':'ASIANPAINT.NS',
    'MARUTI':'MARUTI.NS','NTPC':'NTPC.NS','TITAN':'TITAN.NS',
    'SUNPHARMA':'SUNPHARMA.NS','ULTRACEMCO':'ULTRACEMCO.NS','WIPRO':'WIPRO.NS',
    'BAJAJFINSV':'BAJAJFINSV.NS','ONGC':'ONGC.NS','TECHM':'TECHM.NS',
    'NESTLEIND':'NESTLEIND.NS','ADANIENT':'ADANIENT.NS','POWERGRID':'POWERGRID.NS',
    'HCLTECH':'HCLTECH.NS','TATAMOTORS':'TATAMOTORS.NS','JSWSTEEL':'JSWSTEEL.NS',
    'TATASTEEL':'TATASTEEL.NS','INDUSINDBK':'INDUSINDBK.NS','DRREDDY':'DRREDDY.NS',
    'CIPLA':'CIPLA.NS','DIVISLAB':'DIVISLAB.NS','EICHERMOT':'EICHERMOT.NS',
    'COALINDIA':'COALINDIA.NS','BPCL':'BPCL.NS','GRASIM':'GRASIM.NS',
    'HEROMOTOCO':'HEROMOTOCO.NS','HINDALCO':'HINDALCO.NS','BRITANNIA':'BRITANNIA.NS',
    'APOLLOHOSP':'APOLLOHOSP.NS','LT':'LT.NS','AXISBANK':'AXISBANK.NS',
    'TATACONSUM':'TATACONSUM.NS','SBILIFE':'SBILIFE.NS','HDFCLIFE':'HDFCLIFE.NS',
    'MM':'M&M.NS','VEDL':'VEDL.NS','LTIM':'LTIM.NS','ADANIPORTS':'ADANIPORTS.NS',
    'UPL':'UPL.NS','BAJAJ-AUTO':'BAJAJ-AUTO.NS','GODREJCP':'GODREJCP.NS',
    'DABUR':'DABUR.NS','MARICO':'MARICO.NS','COLPAL':'COLPAL.NS',
    'PIDILITIND':'PIDILITIND.NS','AMBUJACEM':'AMBUJACEM.NS','ACC':'ACC.NS',
    'BANKBARODA':'BANKBARODA.NS','PNB':'PNB.NS','CANBK':'CANBK.NS',
    'IDFCFIRSTB':'IDFCFIRSTB.NS','FEDERALBNK':'FEDERALBNK.NS',
    'MUTHOOTFIN':'MUTHOOTFIN.NS','CHOLAFIN':'CHOLAFIN.NS','RECLTD':'RECLTD.NS',
    'PFC':'PFC.NS','IRCTC':'IRCTC.NS','ZOMATO':'ZOMATO.NS','NYKAA':'NYKAA.NS',
    'TATAPOWER':'TATAPOWER.NS','LUPIN':'LUPIN.NS','HAVELLS':'HAVELLS.NS',
    'INDIGO':'INDIGO.NS','ADANIGREEN':'ADANIGREEN.NS','TRENT':'TRENT.NS',
    'DMART':'DMART.NS','OFSS':'OFSS.NS','MPHASIS':'MPHASIS.NS',
    'PERSISTENT':'PERSISTENT.NS','COFORGE':'COFORGE.NS','DIXON':'DIXON.NS',
    'NIFTY50':'^NSEI','BANKNIFTY':'^NSEBANK','SENSEX':'^BSESN',
}

# ── Indicators ──────────────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return round(float(100 - (100 / (1 + rs)).iloc[-1]), 2)

def calculate_macd(prices):
    ema12  = prices.ewm(span=12, adjust=False).mean()
    ema26  = prices.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])

def calculate_bb(prices, period=20):
    sma   = prices.rolling(period).mean()
    std   = prices.rolling(period).std()
    return float((sma + 2*std).iloc[-1]), float(sma.iloc[-1]), float((sma - 2*std).iloc[-1])

def calculate_ma(prices):
    return float(prices.rolling(20).mean().iloc[-1]), float(prices.rolling(50).mean().iloc[-1])

def get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50):
    buy = sell = 0
    if rsi < 35:              buy  += 2
    elif rsi > 65:            sell += 2
    if macd_val > macd_sig:   buy  += 1
    else:                     sell += 1
    if price < bb_lower:      buy  += 2
    elif price > bb_upper:    sell += 2
    if ma20 > ma50:           buy  += 1
    else:                     sell += 1
    return 'BUY' if buy >= 3 else ('SELL' if sell >= 3 else 'HOLD')

def get_confidence(rsi, macd_val, macd_sig, price, bb_upper, bb_lower):
    score = 50
    if rsi < 35 or rsi > 65:                   score += 15
    if abs(macd_val - macd_sig) > 0.5:         score += 10
    if price < bb_lower or price > bb_upper:   score += 15
    return min(int(score), 95)

# ── /api/stocks ─────────────────────────────────────────────────

@app.route('/api/stocks')
def get_stocks():
    result = []
    for name, ticker in STOCKS.items():
        try:
            df     = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if df.empty or len(df) < 30: continue
            prices = df['Close'].squeeze()
            price  = float(prices.iloc[-1])
            prev   = float(prices.iloc[-2])
            change = round((price - prev) / prev * 100, 2)
            rsi    = calculate_rsi(prices)
            mv, ms = calculate_macd(prices)
            bu,_,bl= calculate_bb(prices)
            ma20,ma50 = calculate_ma(prices)
            result.append({
                'name': name, 'price': round(price,2), 'change': change,
                'rsi': rsi, 'signal': get_signal(rsi,mv,ms,price,bu,bl,ma20,ma50),
                'confidence': get_confidence(rsi,mv,ms,price,bu,bl),
                'macd': round(mv,3), 'macd_signal': round(ms,3),
                'bb_upper': round(bu,2), 'bb_lower': round(bl,2),
            })
        except Exception as e:
            print(f"Error {name}: {e}")
    return jsonify(result)

# ── /api/backtest ────────────────────────────────────────────────

@app.route('/api/backtest/<ticker>/<strategy>/<period>')
def backtest(ticker, strategy, period):
    symbol     = STOCKS.get(ticker, ticker + '.NS')
    period_map = {'3M':'3mo','6M':'6mo','1Y':'1y','3Y':'3y','5Y':'5y'}
    yf_period  = period_map.get(period, '1y')

    try:
        df     = yf.download(symbol, period=yf_period, interval='1d', progress=False)
        if df.empty or len(df) < 60:
            return jsonify({'error': 'Not enough historical data'})

        close  = df['Close'].squeeze().reset_index(drop=True)
        high   = df['High'].squeeze().reset_index(drop=True)
        low    = df['Low'].squeeze().reset_index(drop=True)
        volume = df['Volume'].squeeze().reset_index(drop=True)
        dates  = [str(d.date()) for d in df.index]

        def get_sig(i):
            if i < 52: return 'HOLD'
            c = close.iloc[:i]
            h = high.iloc[:i]
            l = low.iloc[:i]
            v = volume.iloc[:i]
            price   = float(c.iloc[-1])
            rsi     = calculate_rsi(c)
            mv, ms  = calculate_macd(c)
            bu,bm,bl= calculate_bb(c)
            ma20,ma50=calculate_ma(c)
            ema9    = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21   = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
            ema50   = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
            ema200  = float(c.ewm(span=min(200,len(c)-1), adjust=False).mean().iloc[-1])
            # Stochastic
            low14   = float(l.iloc[-14:].min())
            high14  = float(h.iloc[-14:].max())
            stoch   = (price-low14)/(high14-low14)*100 if high14!=low14 else 50
            # ATR
            tr_list = [max(float(h.iloc[j])-float(l.iloc[j]),
                          abs(float(h.iloc[j])-float(c.iloc[j-1])),
                          abs(float(l.iloc[j])-float(c.iloc[j-1]))) for j in range(-14,0)]
            atr     = float(np.mean(tr_list))
            # ROC
            roc     = (price - float(c.iloc[-10]))/float(c.iloc[-10])*100 if len(c)>=10 else 0
            # Volume
            vol_avg = float(v.iloc[-20:].mean())
            vol_now = float(v.iloc[-1])
            vsurge  = vol_now > vol_avg * 1.5
            # Williams %R
            willr   = (high14-price)/(high14-low14)*-100 if high14!=low14 else -50
            # CCI
            tp      = (price + float(h.iloc[-1]) + float(l.iloc[-1])) / 3
            tp_ser  = [(float(c.iloc[j])+float(h.iloc[j])+float(l.iloc[j]))/3 for j in range(-20,0)]
            tp_mean = np.mean(tp_ser)
            tp_mad  = np.mean([abs(x-tp_mean) for x in tp_ser])
            cci     = (tp-tp_mean)/(0.015*tp_mad) if tp_mad>0 else 0
            # 20-day high/low
            high20  = float(h.iloc[-20:].max())
            low20   = float(l.iloc[-20:].min())

            s = {
                'rsi':             'BUY' if rsi<30 else ('SELL' if rsi>70 else 'HOLD'),
                'macd':            'BUY' if mv>ms else 'SELL',
                'bb':              'BUY' if price<bl else ('SELL' if price>bu else 'HOLD'),
                'ma_cross':        'BUY' if ma20>ma50 else 'SELL',
                'ema_cross':       'BUY' if ema9>ema21 else 'SELL',
                'ema_trend':       'BUY' if price>ema200 and ema50>ema200 else ('SELL' if price<ema200 else 'HOLD'),
                'momentum':        'BUY' if roc>2 and vsurge else ('SELL' if roc<-2 and vsurge else 'HOLD'),
                'stochastic':      'BUY' if stoch<20 else ('SELL' if stoch>80 else 'HOLD'),
                'rsi_macd':        'BUY' if rsi<45 and mv>ms else ('SELL' if rsi>55 and mv<ms else 'HOLD'),
                'bb_rsi':          'BUY' if price<bl and rsi<40 else ('SELL' if price>bu and rsi>60 else 'HOLD'),
                'triple_ema':      'BUY' if ema9>ema21>ema50 else ('SELL' if ema9<ema21<ema50 else 'HOLD'),
                'willr':           'BUY' if willr<-80 else ('SELL' if willr>-20 else 'HOLD'),
                'cci':             'BUY' if cci<-100 else ('SELL' if cci>100 else 'HOLD'),
                'volume_breakout': 'BUY' if vsurge and roc>1 and price>ma20 else ('SELL' if vsurge and roc<-1 and price<ma20 else 'HOLD'),
                'supertrend':      'BUY' if price>bm+atr else ('SELL' if price<bm-atr else 'HOLD'),
                'mean_reversion':  'BUY' if price<bl and rsi<35 and stoch<25 else ('SELL' if price>bu and rsi>65 and stoch>75 else 'HOLD'),
                'trend_following': 'BUY' if ema9>ema21 and mv>ms and price>ema50 else ('SELL' if ema9<ema21 and mv<ms and price<ema50 else 'HOLD'),
                'breakout':        'BUY' if price>=high20*0.995 and vsurge else ('SELL' if price<=low20*1.005 and vsurge else 'HOLD'),
                'combined':        get_signal(rsi,mv,ms,price,bu,bl,ma20,ma50),
            }
            return s.get(strategy, 'HOLD')

        # ── Simulate trades with stop loss ──────────────────────
        trades   = []
        position = None
        entry_i  = None
        sl_pct   = 0.05  # 5% stop loss

        for i in range(52, len(close)):
            sig   = get_sig(i)
            price = float(close.iloc[i])
            date  = dates[i]

            if position is None:
                if sig == 'BUY':
                    position = price
                    entry_i  = i
            else:
                stop_loss = position * (1 - sl_pct)
                if price <= stop_loss or sig == 'SELL':
                    ret = round((price - position) / position * 100, 2)
                    trades.append({
                        'entry':      round(position, 2),
                        'exit':       round(price, 2),
                        'return':     ret,
                        'entry_date': dates[entry_i],
                        'exit_date':  date,
                        'days':       i - entry_i,
                        'exit_reason':'Stop Loss' if price <= stop_loss else 'Signal',
                    })
                    position = None
                    entry_i  = None

        if len(trades) < 2:
            return jsonify({'error': f'Only {len(trades)} trade(s) generated. Try a longer period or different strategy.'})

        returns  = [t['return'] for t in trades]
        wins     = [r for r in returns if r > 0]
        losses   = [r for r in returns if r <= 0]

        total_return  = round(sum(returns), 2)
        win_rate      = round(len(wins)/len(returns)*100, 1)
        avg_win       = round(np.mean(wins), 2)   if wins   else 0
        avg_loss      = round(abs(np.mean(losses)),2) if losses else 0
        profit_factor = round(avg_win/avg_loss, 2) if avg_loss > 0 else round(avg_win, 2)
        sharpe        = round(np.mean(returns)/np.std(returns)*np.sqrt(252), 2) if np.std(returns) > 0 else 0
        expectancy    = round((win_rate/100*avg_win) - ((1-win_rate/100)*avg_loss), 2)
        best_trade    = round(max(returns), 2)
        worst_trade   = round(min(returns), 2)
        avg_holding   = round(np.mean([t['days'] for t in trades]), 1)

        # Max drawdown (proper calculation on equity curve)
        equity = [100.0]
        for r in returns:
            equity.append(round(equity[-1] * (1 + r/100), 2))
        peak   = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak: peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd: max_dd = dd
        max_dd = round(max_dd, 2)

        # Consecutive wins/losses
        cw = cl = mcw = mcl = 0
        for r in returns:
            if r > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
            else:     cl += 1; cw = 0; mcl = max(mcl, cl)

        # Monthly returns
        monthly = [0.0] * 12
        chunk   = max(1, len(returns)//12)
        for idx in range(12):
            slice_ = returns[idx*chunk:(idx+1)*chunk]
            if slice_: monthly[idx] = round(sum(slice_), 2)

        return jsonify({
            'total_return':       total_return,
            'win_rate':           win_rate,
            'total_trades':       len(trades),
            'sharpe':             sharpe,
            'max_drawdown':       max_dd,
            'profit_factor':      profit_factor,
            'avg_win':            avg_win,
            'avg_loss':           avg_loss,
            'expectancy':         expectancy,
            'best_trade':         best_trade,
            'worst_trade':        worst_trade,
            'avg_holding_days':   avg_holding,
            'max_consec_wins':    mcw,
            'max_consec_losses':  mcl,
            'monthly_returns':    monthly,
            'equity_curve':       equity,
            'trades':             trades[-20:],
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ── /api/chart ───────────────────────────────────────────────────

@app.route('/api/chart/<ticker>')
def chart_data(ticker):
    symbol = STOCKS.get(ticker, ticker + '.NS')
    try:
        df     = yf.download(symbol, period='3y', interval='1d', progress=False)
        prices = df['Close'].squeeze()
        volumes= df['Volume'].squeeze()
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd_l = ema12 - ema26
        sig_l  = macd_l.ewm(span=9, adjust=False).mean()
        hist_l = macd_l - sig_l
        rsi_s  = []
        for i in range(14, len(prices)):
            rsi_s.append(round(calculate_rsi(prices.iloc[:i+1]), 2))
        return jsonify({
            'dates':       [str(d.date()) for d in df.index],
            'prices':      [round(float(p),2) for p in prices],
            'volumes':     [int(v) for v in volumes],
            'rsi':         [None]*14 + rsi_s,
            'macd':        [round(float(m),3) for m in macd_l],
            'macd_signal': [round(float(s),3) for s in sig_l],
            'macd_hist':   [round(float(h),3) for h in hist_l],
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ── /api/options ─────────────────────────────────────────────────

@app.route('/api/options/<ticker>')
def options_data(ticker):
    INDEX_MAP  = {'NIFTY50':'NIFTY','BANKNIFTY':'BANKNIFTY','FINNIFTY':'FINNIFTY'}
    is_index   = ticker in INDEX_MAP
    nse_symbol = INDEX_MAP.get(ticker, ticker)
    yf_symbol  = STOCKS.get(ticker, ticker + '.NS')

    headers = {
        'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-IN,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer':         'https://www.nseindia.com/option-chain',
        'Connection':      'keep-alive',
    }

    try:
        session = req.Session()
        session.get('https://www.nseindia.com', headers=headers, timeout=8)
        time.sleep(2)
        session.get('https://www.nseindia.com/option-chain', headers=headers, timeout=8)
        time.sleep(2)
        url = f'https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}' if is_index else f'https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}'
        res = session.get(url, headers=headers, timeout=12)

        if res.status_code == 200 and len(res.text) > 200:
            data = res.json()
            if 'records' in data:
                records  = data['records']
                cp       = float(records['underlyingValue'])
                expiries = records['expiryDates']
                expiry   = expiries[0]
                calls_data, puts_data = [], []
                for item in records['data']:
                    if item.get('expiryDate') != expiry: continue
                    s = item['strikePrice']
                    if abs(s - cp) > cp * 0.15: continue
                    if 'CE' in item:
                        ce = item['CE']
                        calls_data.append({'strike':s,'lastPrice':ce.get('lastPrice',0),'bid':ce.get('bidprice',0),'ask':ce.get('askPrice',0),'volume':ce.get('totalTradedVolume',0),'openInterest':ce.get('openInterest',0),'iv':round(ce.get('impliedVolatility',0),1),'inTheMoney':s<cp,'changeinOI':ce.get('changeinOpenInterest',0)})
                    if 'PE' in item:
                        pe = item['PE']
                        puts_data.append({'strike':s,'lastPrice':pe.get('lastPrice',0),'bid':pe.get('bidprice',0),'ask':pe.get('askPrice',0),'volume':pe.get('totalTradedVolume',0),'openInterest':pe.get('openInterest',0),'iv':round(pe.get('impliedVolatility',0),1),'inTheMoney':s>cp,'changeinOI':pe.get('changeinOpenInterest',0)})

                strikes = sorted(set([c['strike'] for c in calls_data]+[p['strike'] for p in puts_data]))
                mp, min_p = None, float('inf')
                for s in strikes:
                    t = sum(max(0,s-c['strike'])*c['openInterest'] for c in calls_data) + sum(max(0,p['strike']-s)*p['openInterest'] for p in puts_data)
                    if t < min_p: min_p = t; mp = s

                mco = max((c['openInterest'] for c in calls_data),default=0)
                mpo = max((p['openInterest'] for p in puts_data), default=0)
                sigs = []
                for c in calls_data:
                    if c['openInterest']>=mco*0.7 and c['strike']>cp:
                        sigs.append({'type':'CALL','strike':c['strike'],'premium':c['lastPrice'],'oi':c['openInterest'],'iv':c['iv'],'signal':'RESISTANCE','reason':f"Highest call OI at {c['strike']}"})
                for p in puts_data:
                    if p['openInterest']>=mpo*0.7 and p['strike']<cp:
                        sigs.append({'type':'PUT','strike':p['strike'],'premium':p['lastPrice'],'oi':p['openInterest'],'iv':p['iv'],'signal':'SUPPORT','reason':f"Highest put OI at {p['strike']}"})

                return jsonify({'ticker':ticker,'currentPrice':cp,'expiry':expiry,'expiries':expiries[:5],'calls':calls_data,'puts':puts_data,'maxPain':mp,'fnoSignals':sigs[:8],'source':'NSE'})
    except Exception as e:
        print(f"NSE failed: {e}")

    # yfinance fallback
    try:
        s     = yf.Ticker(yf_symbol)
        hist  = s.history(period='2d')
        if hist.empty: return jsonify({'error':'No price data'})
        cp    = round(float(hist['Close'].iloc[-1]),2)
        exps  = s.options
        if not exps:
            return jsonify({'ticker':ticker,'currentPrice':cp,'expiry':'N/A','expiries':[],'calls':[],'puts':[],'maxPain':None,'fnoSignals':[],'source':'yfinance','warning':'Options not available. Works better when backend runs locally.'})
        expiry = exps[0]
        chain  = s.option_chain(expiry)
        def clean(df):
            r=[]
            for _,row in df.iterrows():
                try: r.append({'strike':round(float(row['strike']),2),'lastPrice':round(float(row['lastPrice']),2),'bid':round(float(row.get('bid',0)),2),'ask':round(float(row.get('ask',0)),2),'volume':int(row.get('volume',0)) if str(row.get('volume','nan'))!='nan' else 0,'openInterest':int(row.get('openInterest',0)) if str(row.get('openInterest','nan'))!='nan' else 0,'iv':round(float(row.get('impliedVolatility',0))*100,1),'inTheMoney':bool(row.get('inTheMoney',False)),'changeinOI':0})
                except: continue
            return r
        low=cp*0.85; high=cp*1.15
        calls_data = clean(chain.calls[(chain.calls['strike']>=low)&(chain.calls['strike']<=high)])
        puts_data  = clean(chain.puts[(chain.puts['strike']>=low)&(chain.puts['strike']<=high)])
        strikes = sorted(set([c['strike'] for c in calls_data]+[p['strike'] for p in puts_data]))
        mp, min_p = None, float('inf')
        for sv in strikes:
            t = sum(max(0,sv-c['strike'])*c['openInterest'] for c in calls_data)+sum(max(0,p['strike']-sv)*p['openInterest'] for p in puts_data)
            if t<min_p: min_p=t; mp=sv
        return jsonify({'ticker':ticker,'currentPrice':cp,'expiry':expiry,'expiries':exps[:5],'calls':calls_data,'puts':puts_data,'maxPain':mp,'fnoSignals':[],'source':'yfinance'})
    except Exception as e:
        return jsonify({'error':f'Options unavailable: {str(e)}'})

# ── /api/screener ────────────────────────────────────────────────

@app.route('/api/screener')
def screener():
    result = []
    for name, ticker in STOCKS.items():
        try:
            df = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if df.empty or len(df)<30: continue
            prices = df['Close'].squeeze()
            price  = float(prices.iloc[-1])
            prev   = float(prices.iloc[-2])
            change = round((price-prev)/prev*100, 2)
            rsi    = calculate_rsi(prices)
            mv, ms = calculate_macd(prices)
            bu,_,bl= calculate_bb(prices)
            ma20,ma50=calculate_ma(prices)
            va = float(df['Volume'].squeeze().rolling(20).mean().iloc[-1])
            vn = float(df['Volume'].squeeze().iloc[-1])
            result.append({
                'name':name,'price':round(price,2),'change':change,'rsi':rsi,
                'signal':get_signal(rsi,mv,ms,price,bu,bl,ma20,ma50),
                'confidence':get_confidence(rsi,mv,ms,price,bu,bl),
                'macd':round(mv,3),'macd_signal':round(ms,3),
                'bb_upper':round(bu,2),'bb_lower':round(bl,2),
                'ma20':round(ma20,2),'ma50':round(ma50,2),
                'vol_ratio':round(vn/va,2) if va>0 else 1.0,
            })
        except: continue
    return jsonify(result)

# ── /api/alerts/check ────────────────────────────────────────────

@app.route('/api/alerts/check')
def check_alerts():
    triggered = []
    for name, ticker in list(STOCKS.items())[:20]:
        try:
            df = yf.download(ticker, period='1mo', interval='1d', progress=False)
            if df.empty or len(df)<20: continue
            prices = df['Close'].squeeze()
            price  = float(prices.iloc[-1])
            rsi    = calculate_rsi(prices)
            mv, ms = calculate_macd(prices)
            bu,_,bl= calculate_bb(prices)
            ma20,ma50=calculate_ma(prices)
            signal = get_signal(rsi,mv,ms,price,bu,bl,ma20,ma50)
            if signal in ['BUY','SELL']:
                triggered.append({
                    'stock':name,'signal':signal,'price':round(price,2),'rsi':rsi,
                    'time':pd.Timestamp.now().strftime('%H:%M'),
                    'date':pd.Timestamp.now().strftime('%d %b %Y'),
                })
        except: continue
    return jsonify(triggered)

# ── Zerodha Auth ─────────────────────────────────────────────────

@app.route('/api/broker/login-url')
def broker_login_url():
    try:
        kite = KiteConnect(api_key=API_KEY)
        url  = kite.login_url()
        return jsonify({'url': url})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/callback')
def broker_callback():
    request_token = request.args.get('request_token')
    if not request_token:
        return jsonify({'error': 'No request token received'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data['access_token']
        user_id      = data['user_id']
        kite_sessions[user_id] = access_token
        # Redirect to frontend with token
        from flask import redirect
        return redirect(f'http://localhost:5173/broker?token={access_token}&user_id={user_id}')
    except Exception as e:
        return redirect(f'http://localhost:5173/broker?error={str(e)}')


@app.route('/api/broker/profile')
def broker_profile():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        profile = kite.profile()
        margins = kite.margins()
        return jsonify({
            'user_id':    profile['user_id'],
            'user_name':  profile['user_name'],
            'email':      profile['email'],
            'broker':     profile['broker'],
            'equity_margin': margins.get('equity', {}).get('available', {}).get('live_balance', 0),
            'commodity_margin': margins.get('commodity', {}).get('available', {}).get('live_balance', 0),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/positions')
def broker_positions():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        positions = kite.positions()
        net = positions.get('net', [])
        result = []
        for p in net:
            if p['quantity'] == 0:
                continue
            result.append({
                'symbol':       p['tradingsymbol'],
                'exchange':     p['exchange'],
                'product':      p['product'],
                'quantity':     p['quantity'],
                'avg_price':    round(p['average_price'], 2),
                'ltp':          round(p['last_price'], 2),
                'pnl':          round(p['pnl'], 2),
                'pnl_pct':      round((p['pnl'] / (p['average_price'] * abs(p['quantity'])) * 100), 2) if p['average_price'] > 0 else 0,
                'value':        round(p['last_price'] * abs(p['quantity']), 2),
                'type':         'LONG' if p['quantity'] > 0 else 'SHORT',
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/pnl')
def broker_pnl():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        positions = kite.positions()
        net = positions.get('net', [])
        total_pnl      = sum(p['pnl'] for p in net)
        realised_pnl   = sum(p['realised'] for p in net)
        unrealised_pnl = sum(p['unrealised'] for p in net)
        winners        = [p for p in net if p['pnl'] > 0]
        losers         = [p for p in net if p['pnl'] < 0]
        return jsonify({
            'total_pnl':      round(total_pnl, 2),
            'realised_pnl':   round(realised_pnl, 2),
            'unrealised_pnl': round(unrealised_pnl, 2),
            'total_positions':len(net),
            'winners':        len(winners),
            'losers':         len(losers),
            'win_rate':       round(len(winners)/len(net)*100, 1) if net else 0,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/orders')
def broker_orders():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        orders = kite.orders()
        result = []
        for o in orders:
            result.append({
                'order_id':       o['order_id'],
                'symbol':         o['tradingsymbol'],
                'exchange':       o['exchange'],
                'transaction':    o['transaction_type'],
                'order_type':     o['order_type'],
                'product':        o['product'],
                'quantity':       o['quantity'],
                'price':          round(o['price'], 2),
                'avg_price':      round(o['average_price'], 2),
                'status':         o['status'],
                'placed_at':      str(o['order_timestamp']),
            })
        return jsonify(list(reversed(result)))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/trades')
def broker_trades():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        trades = kite.trades()
        result = []
        for t in trades:
            result.append({
                'trade_id':    t['trade_id'],
                'order_id':    t['order_id'],
                'symbol':      t['tradingsymbol'],
                'exchange':    t['exchange'],
                'transaction': t['transaction_type'],
                'product':     t['product'],
                'quantity':    t['quantity'],
                'price':       round(t['price'], 2),
                'filled_at':   str(t['fill_timestamp']),
            })
        return jsonify(list(reversed(result)))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/place-order', methods=['POST'])
def place_order():
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        data       = request.json
        kite       = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        order_id   = kite.place_order(
            tradingsymbol = data['symbol'],
            exchange      = data.get('exchange', 'NSE'),
            transaction_type = data['transaction'],
            quantity      = int(data['quantity']),
            order_type    = data.get('order_type', 'MARKET'),
            product       = data.get('product', 'MIS'),
            price         = data.get('price', 0),
            variety       = data.get('variety', 'regular'),
        )
        return jsonify({'success': True, 'order_id': order_id})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/broker/cancel-order/<order_id>', methods=['DELETE'])
def cancel_order(order_id):
    token = request.headers.get('X-Kite-Token')
    if not token:
        return jsonify({'error': 'Not authenticated'})
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(token)
        kite.cancel_order(variety='regular', order_id=order_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)