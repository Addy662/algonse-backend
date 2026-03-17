from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# NSE stocks list
STOCKS = {
    # NIFTY 50
    'RELIANCE':    'RELIANCE.NS',
    'HDFCBANK':    'HDFCBANK.NS',
    'TCS':         'TCS.NS',
    'INFY':        'INFY.NS',
    'ICICIBANK':   'ICICIBANK.NS',
    'HINDUNILVR':  'HINDUNILVR.NS',
    'ITC':         'ITC.NS',
    'SBIN':        'SBIN.NS',
    'BHARTIARTL':  'BHARTIARTL.NS',
    'KOTAKBANK':   'KOTAKBANK.NS',
    'BAJFINANCE':  'BAJFINANCE.NS',
    'ASIANPAINT':  'ASIANPAINT.NS',
    'MARUTI':      'MARUTI.NS',
    'NTPC':        'NTPC.NS',
    'TITAN':       'TITAN.NS',
    'SUNPHARMA':   'SUNPHARMA.NS',
    'ULTRACEMCO':  'ULTRACEMCO.NS',
    'WIPRO':       'WIPRO.NS',
    'BAJAJFINSV':  'BAJAJFINSV.NS',
    'ONGC':        'ONGC.NS',
    'TECHM':       'TECHM.NS',
    'NESTLEIND':   'NESTLEIND.NS',
    'ADANIENT':    'ADANIENT.NS',
    'POWERGRID':   'POWERGRID.NS',
    'HCLTECH':     'HCLTECH.NS',
    'TATAMOTORS':  'TATAMOTORS.NS',
    'JSWSTEEL':    'JSWSTEEL.NS',
    'TATASTEEL':   'TATASTEEL.NS',
    'INDUSINDBK':  'INDUSINDBK.NS',
    'DRREDDY':     'DRREDDY.NS',
    'CIPLA':       'CIPLA.NS',
    'DIVISLAB':    'DIVISLAB.NS',
    'EICHERMOT':   'EICHERMOT.NS',
    'COALINDIA':   'COALINDIA.NS',
    'BPCL':        'BPCL.NS',
    'GRASIM':      'GRASIM.NS',
    'HEROMOTOCO':  'HEROMOTOCO.NS',
    'HINDALCO':    'HINDALCO.NS',
    'BRITANNIA':   'BRITANNIA.NS',
    'APOLLOHOSP':  'APOLLOHOSP.NS',
    'LT':          'LT.NS',
    'AXISBANK':    'AXISBANK.NS',
    'TATACONSUM':  'TATACONSUM.NS',
    'SBILIFE':     'SBILIFE.NS',
    'HDFCLIFE':    'HDFCLIFE.NS',
    'MM':          'M&M.NS',
    'VEDL':        'VEDL.NS',
    'LTIM':        'LTIM.NS',
    'ADANIPORTS':  'ADANIPORTS.NS',
    'UPL':         'UPL.NS',
    # NIFTY NEXT 50
    'BAJAJ-AUTO':  'BAJAJ-AUTO.NS',
    'GODREJCP':    'GODREJCP.NS',
    'DABUR':       'DABUR.NS',
    'MARICO':      'MARICO.NS',
    'BERGEPAINT':  'BERGEPAINT.NS',
    'COLPAL':      'COLPAL.NS',
    'PIDILITIND':  'PIDILITIND.NS',
    'AMBUJACEM':   'AMBUJACEM.NS',
    'ACC':         'ACC.NS',
    'BANKBARODA':  'BANKBARODA.NS',
    'PNB':         'PNB.NS',
    'CANBK':       'CANBK.NS',
    'IDFCFIRSTB':  'IDFCFIRSTB.NS',
    'FEDERALBNK':  'FEDERALBNK.NS',
    'MUTHOOTFIN':  'MUTHOOTFIN.NS',
    'CHOLAFIN':    'CHOLAFIN.NS',
    'RECLTD':      'RECLTD.NS',
    'PFC':         'PFC.NS',
    'IRCTC':       'IRCTC.NS',
    'ZOMATO':      'ZOMATO.NS',
    'NYKAA':       'NYKAA.NS',
    'DELHIVERY':   'DELHIVERY.NS',
    'TATAPOWER':   'TATAPOWER.NS',
    'TORNTPHARM':  'TORNTPHARM.NS',
    'LUPIN':       'LUPIN.NS',
    'AUROPHARMA':  'AUROPHARMA.NS',
    'BIOCON':      'BIOCON.NS',
    'MCDOWELL-N':  'MCDOWELL-N.NS',
    'HAVELLS':     'HAVELLS.NS',
    'VOLTAS':      'VOLTAS.NS',
    'INDIGO':      'INDIGO.NS',
    'ADANIGREEN':  'ADANIGREEN.NS',
    'ADANIPOWER':  'ADANIPOWER.NS',
    'ADANITRANS':  'ADANITRANS.NS',
    'TRENT':       'TRENT.NS',
    'NAUKRI':      'NAUKRI.NS',
    'POLICYBZR':   'POLICYBZR.NS',
    'PAYTM':       'PAYTM.NS',
    'DMART':       'DMART.NS',
    'OFSS':        'OFSS.NS',
    'MPHASIS':     'MPHASIS.NS',
    'PERSISTENT':  'PERSISTENT.NS',
    'COFORGE':     'COFORGE.NS',
    'LTTS':        'LTTS.NS',
    'KPIT':        'KPIT.NS',
    'DIXON':       'DIXON.NS',
    'DEEPAKNTR':   'DEEPAKNTR.NS',
    'PIIND':       'PIIND.NS',
    'ESCORTS':     'ESCORTS.NS',
    'BALKRISIND':  'BALKRISIND.NS',
    'CUMMINSIND':  'CUMMINSIND.NS',
    'THERMAX':     'THERMAX.NS',
    # INDICES
    'NIFTY50':     '^NSEI',
    'BANKNIFTY':   '^NSEBANK',
    'SENSEX':      '^BSESN',
}

# ── Indicator calculations ──────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)

def calculate_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])

def calculate_bb(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

def calculate_ma(prices):
    ma20 = float(prices.rolling(20).mean().iloc[-1])
    ma50 = float(prices.rolling(50).mean().iloc[-1])
    return ma20, ma50

def get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50):
    buy_signals = 0
    sell_signals = 0

    # RSI
    if rsi < 35:
        buy_signals += 2
    elif rsi > 65:
        sell_signals += 2

    # MACD
    if macd_val > macd_sig:
        buy_signals += 1
    else:
        sell_signals += 1

    # Bollinger Bands
    if price < bb_lower:
        buy_signals += 2
    elif price > bb_upper:
        sell_signals += 2

    # Moving Average
    if ma20 > ma50:
        buy_signals += 1
    else:
        sell_signals += 1

    if buy_signals >= 3:
        return 'BUY'
    elif sell_signals >= 3:
        return 'SELL'
    else:
        return 'HOLD'

def get_confidence(rsi, macd_val, macd_sig, price, bb_upper, bb_lower):
    score = 50
    if rsi < 35: score += 15
    elif rsi > 65: score += 15
    if abs(macd_val - macd_sig) > 0.5: score += 10
    if price < bb_lower or price > bb_upper: score += 15
    return min(int(score), 95)

# ── Routes ──────────────────────────────────────────────────────

@app.route('/api/stocks')
def get_stocks():
    result = []
    for name, ticker in STOCKS.items():
        try:
            df = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if df.empty:
                continue

            prices = df['Close'].squeeze()
            price = float(prices.iloc[-1])
            prev_price = float(prices.iloc[-2])
            change_pct = round(((price - prev_price) / prev_price) * 100, 2)

            rsi = calculate_rsi(prices)
            macd_val, macd_sig = calculate_macd(prices)
            bb_upper, bb_mid, bb_lower = calculate_bb(prices)
            ma20, ma50 = calculate_ma(prices)

            signal = get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50)
            confidence = get_confidence(rsi, macd_val, macd_sig, price, bb_upper, bb_lower)

            result.append({
                'name': name,
                'price': round(price, 2),
                'change': change_pct,
                'rsi': rsi,
                'signal': signal,
                'confidence': confidence,
                'macd': round(macd_val, 3),
                'macd_signal': round(macd_sig, 3),
                'bb_upper': round(bb_upper, 2),
                'bb_lower': round(bb_lower, 2),
            })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue

    return jsonify(result)


@app.route('/api/backtest/<ticker>/<strategy>/<period>')
def backtest(ticker, strategy, period):
    symbol   = STOCKS.get(ticker, ticker + '.NS')
    period_map = {'3M':'3mo','6M':'6mo','1Y':'1y','3Y':'3y','5Y':'5y'}
    yf_period  = period_map.get(period, '1y')

    try:
        df     = yf.download(symbol, period=yf_period, interval='1d', progress=False)
        prices = df['Close'].squeeze()
        highs  = df['High'].squeeze()
        lows   = df['Low'].squeeze()
        vols   = df['Volume'].squeeze()

        def get_strategy_signal(i, strategy):
            chunk  = prices.iloc[:i]
            if len(chunk) < 50:
                return 'HOLD'

            rsi            = calculate_rsi(chunk)
            macd_val, macd_sig = calculate_macd(chunk)
            bb_upper, bb_mid, bb_lower = calculate_bb(chunk)
            ma20, ma50     = calculate_ma(chunk)
            price          = float(chunk.iloc[-1])

            # EMA values
            ema9  = float(chunk.ewm(span=9).mean().iloc[-1])
            ema21 = float(chunk.ewm(span=21).mean().iloc[-1])
            ema50 = float(chunk.ewm(span=50).mean().iloc[-1])
            ema200= float(chunk.ewm(span=200).mean().iloc[-1]) if len(chunk) >= 200 else ema50

            # Stochastic
            low14  = float(lows.iloc[max(0,i-14):i].min())
            high14 = float(highs.iloc[max(0,i-14):i].max())
            stoch_k = ((price - low14) / (high14 - low14) * 100) if high14 != low14 else 50

            # ATR
            atr_vals = []
            for j in range(max(1,i-14), i):
                tr = max(
                    float(highs.iloc[j]) - float(lows.iloc[j]),
                    abs(float(highs.iloc[j]) - float(prices.iloc[j-1])),
                    abs(float(lows.iloc[j])  - float(prices.iloc[j-1]))
                )
                atr_vals.append(tr)
            atr = np.mean(atr_vals) if atr_vals else price * 0.02

            # Volume
            vol_avg = float(vols.iloc[max(0,i-20):i].mean())
            vol_now = float(vols.iloc[i-1]) if i > 0 else vol_avg
            vol_surge = vol_now > vol_avg * 1.5

            # ROC
            roc = ((price - float(chunk.iloc[-10])) / float(chunk.iloc[-10]) * 100) if len(chunk) >= 10 else 0

            # Williams %R
            willr = ((high14 - price) / (high14 - low14) * -100) if high14 != low14 else -50

            # CCI
            typical = (price + float(highs.iloc[i-1]) + float(lows.iloc[i-1])) / 3
            tp_mean = np.mean([(float(prices.iloc[j]) + float(highs.iloc[j]) + float(lows.iloc[j]))/3 for j in range(max(0,i-20),i)])
            tp_mad  = np.mean([abs((float(prices.iloc[j]) + float(highs.iloc[j]) + float(lows.iloc[j]))/3 - tp_mean) for j in range(max(0,i-20),i)])
            cci     = (typical - tp_mean) / (0.015 * tp_mad) if tp_mad != 0 else 0

            strategies = {
                'rsi':            'BUY' if rsi < 30 else ('SELL' if rsi > 70 else 'HOLD'),
                'macd':           'BUY' if macd_val > macd_sig else 'SELL',
                'bb':             'BUY' if price < bb_lower else ('SELL' if price > bb_upper else 'HOLD'),
                'ma_cross':       'BUY' if ma20 > ma50 else 'SELL',
                'ema_cross':      'BUY' if ema9 > ema21 else 'SELL',
                'ema_trend':      'BUY' if price > ema200 and ema50 > ema200 else ('SELL' if price < ema200 else 'HOLD'),
                'momentum':       'BUY' if roc > 3 and vol_surge else ('SELL' if roc < -3 and vol_surge else 'HOLD'),
                'stochastic':     'BUY' if stoch_k < 20 else ('SELL' if stoch_k > 80 else 'HOLD'),
                'rsi_macd':       'BUY' if rsi < 40 and macd_val > macd_sig else ('SELL' if rsi > 60 and macd_val < macd_sig else 'HOLD'),
                'bb_rsi':         'BUY' if price < bb_lower and rsi < 35 else ('SELL' if price > bb_upper and rsi > 65 else 'HOLD'),
                'triple_ema':     'BUY' if ema9 > ema21 > ema50 else ('SELL' if ema9 < ema21 < ema50 else 'HOLD'),
                'willr':          'BUY' if willr < -80 else ('SELL' if willr > -20 else 'HOLD'),
                'cci':            'BUY' if cci < -100 else ('SELL' if cci > 100 else 'HOLD'),
                'volume_breakout':'BUY' if vol_surge and roc > 1 else ('SELL' if vol_surge and roc < -1 else 'HOLD'),
                'supertrend':     'BUY' if price > bb_mid + atr else ('SELL' if price < bb_mid - atr else 'HOLD'),
                'mean_reversion': 'BUY' if price < bb_lower and rsi < 40 and stoch_k < 25 else ('SELL' if price > bb_upper and rsi > 60 and stoch_k > 75 else 'HOLD'),
                'trend_following':'BUY' if ema9 > ema21 and macd_val > macd_sig and price > ema50 else ('SELL' if ema9 < ema21 and macd_val < macd_sig and price < ema50 else 'HOLD'),
                'breakout':       'BUY' if price > float(highs.iloc[max(0,i-20):i].max()) * 0.99 and vol_surge else ('SELL' if price < float(lows.iloc[max(0,i-20):i].min()) * 1.01 and vol_surge else 'HOLD'),
                'combined':       get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50),
            }
            return strategies.get(strategy, 'HOLD')

        # Simulate trades
        trades   = []
        position = None
        entry_date = None
        for i in range(50, len(prices)):
            sig   = get_strategy_signal(i, strategy)
            price = float(prices.iloc[i])
            date  = str(df.index[i].date())
            if sig == 'BUY' and position is None:
                position   = price
                entry_date = date
            elif sig == 'SELL' and position is not None:
                ret = ((price - position) / position) * 100
                trades.append({
                    'entry': position, 'exit': price,
                    'return': round(ret, 2),
                    'entry_date': entry_date, 'exit_date': date,
                    'days': (df.index[i] - df.index[df.index.get_loc(entry_date) if entry_date in df.index else i]).days if entry_date else 0
                })
                position = None

        if not trades:
            return jsonify({'error': 'Not enough trades generated. Try a longer period or different strategy.'})

        returns     = [t['return'] for t in trades]
        wins        = [r for r in returns if r > 0]
        losses      = [r for r in returns if r <= 0]
        total_return= round(sum(returns), 2)
        win_rate    = round(len(wins) / len(returns) * 100, 1)
        avg_win     = round(np.mean(wins), 2)   if wins   else 0
        avg_loss    = round(abs(np.mean(losses)),2) if losses else 0
        profit_factor = round(avg_win / avg_loss, 2) if avg_loss > 0 else 99
        max_dd      = round(abs(min(returns)), 2) if returns else 0
        sharpe      = round(np.mean(returns) / np.std(returns) * np.sqrt(252), 2) if len(returns) > 1 else 0
        expectancy  = round((win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss), 2)
        best_trade  = round(max(returns), 2)
        worst_trade = round(min(returns), 2)
        avg_holding = round(np.mean([t['days'] for t in trades if t['days'] > 0]), 1) if trades else 0
        consec_wins = 0
        max_consec_wins = 0
        consec_losses = 0
        max_consec_losses = 0
        for r in returns:
            if r > 0:
                consec_wins += 1; max_consec_wins = max(max_consec_wins, consec_wins); consec_losses = 0
            else:
                consec_losses += 1; max_consec_losses = max(max_consec_losses, consec_losses); consec_wins = 0

        # Monthly returns
        monthly = []
        chunk_size = max(1, len(returns) // 12)
        for i in range(0, min(len(returns), 12 * chunk_size), chunk_size):
            monthly.append(round(sum(returns[i:i+chunk_size]), 2))
        while len(monthly) < 12:
            monthly.append(0.0)

        # Equity curve (cumulative)
        equity = [100.0]
        for r in returns:
            equity.append(round(equity[-1] * (1 + r/100), 2))

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
            'max_consec_wins':    max_consec_wins,
            'max_consec_losses':  max_consec_losses,
            'monthly_returns':    monthly[:12],
            'equity_curve':       equity,
            'trades':             trades[-20:],
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/chart/<ticker>')
def chart_data(ticker):
    symbol = STOCKS.get(ticker, ticker + '.NS')
    try:
        df = yf.download(symbol, period='3y', interval='1d', progress=False)
        prices = df['Close'].squeeze()
        volumes = df['Volume'].squeeze()

        # RSI
        rsi_series = []
        for i in range(14, len(prices)):
            chunk = prices.iloc[:i+1]
            rsi_series.append(round(calculate_rsi(chunk), 2))

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        dates = [str(d.date()) for d in df.index]

        return jsonify({
            'dates': dates,
            'prices': [round(float(p), 2) for p in prices],
            'volumes': [int(v) for v in volumes],
            'rsi': [None]*14 + rsi_series,
            'macd': [round(float(m), 3) for m in macd_line],
            'macd_signal': [round(float(s), 3) for s in signal_line],
            'macd_hist': [round(float(h), 3) for h in histogram],
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/options/<ticker>')
def options_data(ticker):
    import requests as req
    import time

    INDEX_MAP = {
        'NIFTY50':    'NIFTY',
        'BANKNIFTY':  'BANKNIFTY',
        'FINNIFTY':   'FINNIFTY',
        'MIDCPNIFTY': 'MIDCPNIFTY',
    }

    is_index  = ticker in INDEX_MAP
    nse_symbol = INDEX_MAP.get(ticker, ticker)

    headers = {
        'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept':          '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer':         'https://www.nseindia.com/option-chain',
        'X-Requested-With':'XMLHttpRequest',
    }

    session = req.Session()
    try:
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        time.sleep(1)
        session.get('https://www.nseindia.com/option-chain', headers=headers, timeout=10)
        time.sleep(1)

        if is_index:
            url = f'https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}'
        else:
            url = f'https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}'

        res  = session.get(url, headers=headers, timeout=15)
        data = res.json()

        if 'records' not in data:
            return jsonify({'error': 'No data from NSE. Market may be closed.'})

        records       = data['records']
        current_price = float(records['underlyingValue'])
        expiries      = records['expiryDates']
        expiry        = expiries[0]

        calls_data = []
        puts_data  = []

        for item in records['data']:
            if item.get('expiryDate') != expiry:
                continue
            strike = item['strikePrice']
            if abs(strike - current_price) > current_price * 0.15:
                continue
            if 'CE' in item:
                ce = item['CE']
                calls_data.append({
                    'strike':       strike,
                    'lastPrice':    ce.get('lastPrice', 0),
                    'bid':          ce.get('bidprice', 0),
                    'ask':          ce.get('askPrice', 0),
                    'volume':       ce.get('totalTradedVolume', 0),
                    'openInterest': ce.get('openInterest', 0),
                    'iv':           round(ce.get('impliedVolatility', 0), 1),
                    'inTheMoney':   strike < current_price,
                    'changeinOI':   ce.get('changeinOpenInterest', 0),
                })
            if 'PE' in item:
                pe = item['PE']
                puts_data.append({
                    'strike':       strike,
                    'lastPrice':    pe.get('lastPrice', 0),
                    'bid':          pe.get('bidprice', 0),
                    'ask':          pe.get('askPrice', 0),
                    'volume':       pe.get('totalTradedVolume', 0),
                    'openInterest': pe.get('openInterest', 0),
                    'iv':           round(pe.get('impliedVolatility', 0), 1),
                    'inTheMoney':   strike > current_price,
                    'changeinOI':   pe.get('changeinOpenInterest', 0),
                })

        strikes      = sorted(set([c['strike'] for c in calls_data] + [p['strike'] for p in puts_data]))
        max_pain_val = None
        min_pain     = float('inf')
        for s in strikes:
            call_pain = sum(max(0, s - c['strike']) * c['openInterest'] for c in calls_data)
            put_pain  = sum(max(0, p['strike'] - s) * p['openInterest'] for p in puts_data)
            total = call_pain + put_pain
            if total < min_pain:
                min_pain     = total
                max_pain_val = s

        max_call_oi  = max((c['openInterest'] for c in calls_data), default=0)
        max_put_oi   = max((p['openInterest'] for p in puts_data),  default=0)
        fno_signals  = []
        for c in calls_data:
            if c['openInterest'] >= max_call_oi * 0.7 and c['strike'] > current_price:
                fno_signals.append({
                    'type': 'CALL', 'strike': c['strike'],
                    'premium': c['lastPrice'], 'oi': c['openInterest'],
                    'iv': c['iv'], 'signal': 'RESISTANCE',
                    'reason': f"Highest call OI — strong resistance at {c['strike']}"
                })
        for p in puts_data:
            if p['openInterest'] >= max_put_oi * 0.7 and p['strike'] < current_price:
                fno_signals.append({
                    'type': 'PUT', 'strike': p['strike'],
                    'premium': p['lastPrice'], 'oi': p['openInterest'],
                    'iv': p['iv'], 'signal': 'SUPPORT',
                    'reason': f"Highest put OI — strong support at {p['strike']}"
                })

        return jsonify({
            'ticker':       ticker,
            'currentPrice': current_price,
            'expiry':       expiry,
            'expiries':     expiries[:5],
            'calls':        calls_data,
            'puts':         puts_data,
            'maxPain':      max_pain_val,
            'fnoSignals':   fno_signals[:8],
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/screener')
def screener():
    result = []
    for name, ticker in STOCKS.items():
        try:
            df = yf.download(ticker, period='3mo', interval='1d', progress=False)
            if df.empty:
                continue
            prices = df['Close'].squeeze()
            if len(prices) < 30:
                continue
            price      = float(prices.iloc[-1])
            prev_price = float(prices.iloc[-2])
            change_pct = round(((price - prev_price) / prev_price) * 100, 2)
            rsi        = calculate_rsi(prices)
            macd_val, macd_sig = calculate_macd(prices)
            bb_upper, bb_mid, bb_lower = calculate_bb(prices)
            ma20, ma50 = calculate_ma(prices)
            signal     = get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50)
            confidence = get_confidence(rsi, macd_val, macd_sig, price, bb_upper, bb_lower)
            vol_avg    = float(df['Volume'].squeeze().rolling(20).mean().iloc[-1])
            vol_today  = float(df['Volume'].squeeze().iloc[-1])
            vol_ratio  = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0
            result.append({
                'name':       name,
                'price':      round(price, 2),
                'change':     change_pct,
                'rsi':        rsi,
                'signal':     signal,
                'confidence': confidence,
                'macd':       round(macd_val, 3),
                'macd_signal':round(macd_sig, 3),
                'bb_upper':   round(bb_upper, 2),
                'bb_lower':   round(bb_lower, 2),
                'ma20':       round(ma20, 2),
                'ma50':       round(ma50, 2),
                'vol_ratio':  vol_ratio,
            })
        except Exception as e:
            continue
    return jsonify(result)

alerts_store = []

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    return jsonify(alerts_store)

@app.route('/api/alerts/check')
def check_alerts():
    triggered = []
    for name, ticker in list(STOCKS.items())[:20]:
        try:
            df     = yf.download(ticker, period='1mo', interval='1d', progress=False)
            if df.empty: continue
            prices = df['Close'].squeeze()
            if len(prices) < 20: continue
            price      = float(prices.iloc[-1])
            rsi        = calculate_rsi(prices)
            macd_val, macd_sig = calculate_macd(prices)
            bb_upper, _, bb_lower = calculate_bb(prices)
            ma20, ma50 = calculate_ma(prices)
            signal     = get_signal(rsi, macd_val, macd_sig, price, bb_upper, bb_lower, ma20, ma50)
            if signal in ['BUY', 'SELL']:
                triggered.append({
                    'stock':   name,
                    'signal':  signal,
                    'price':   round(price, 2),
                    'rsi':     rsi,
                    'time':    pd.Timestamp.now().strftime('%H:%M'),
                    'date':    pd.Timestamp.now().strftime('%d %b %Y'),
                })
        except:
            continue
    return jsonify(triggered)

if __name__ == '__main__':
    app.run(debug=True, port=5000)