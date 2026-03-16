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
    symbol = STOCKS.get(ticker, ticker + '.NS')
    period_map = {'3M': '3mo', '6M': '6mo', '1Y': '1y', '3Y': '3y', '5Y': '5y'}
    yf_period = period_map.get(period, '1y')

    try:
        df = yf.download(symbol, period=yf_period, interval='1d', progress=False)
        prices = df['Close'].squeeze()

        # Generate trades based on strategy
        signals = []
        for i in range(50, len(prices)):
            chunk = prices.iloc[:i]
            rsi = calculate_rsi(chunk)
            macd_val, macd_sig = calculate_macd(chunk)
            bb_upper, _, bb_lower = calculate_bb(chunk)
            ma20, ma50 = calculate_ma(chunk)
            sig = get_signal(rsi, macd_val, macd_sig, float(chunk.iloc[-1]), bb_upper, bb_lower, ma20, ma50)
            signals.append(sig)

        # Simple backtest: buy on BUY, sell on SELL
        trades = []
        position = None
        for i, sig in enumerate(signals):
            idx = i + 50
            p = float(prices.iloc[idx])
            if sig == 'BUY' and position is None:
                position = p
            elif sig == 'SELL' and position is not None:
                ret = ((p - position) / position) * 100
                trades.append(ret)
                position = None

        if not trades:
            return jsonify({'error': 'Not enough data for backtest'})

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        total_return = round(sum(trades), 2)
        win_rate = round(len(wins) / len(trades) * 100, 1)
        avg_win = round(np.mean(wins), 2) if wins else 0
        avg_loss = round(abs(np.mean(losses)), 2) if losses else 0
        profit_factor = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0
        max_dd = round(abs(min(trades)), 2) if trades else 0
        sharpe = round(np.mean(trades) / np.std(trades) * np.sqrt(252), 2) if len(trades) > 1 else 0

        # Monthly returns (simplified)
        monthly = []
        chunk_size = max(1, len(trades) // 12)
        for i in range(0, min(len(trades), 12 * chunk_size), chunk_size):
            monthly.append(round(sum(trades[i:i+chunk_size]), 2))
        while len(monthly) < 12:
            monthly.append(0)

        return jsonify({
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'monthly_returns': monthly[:12],
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
    
    INDEX_MAP = {
        'NIFTY50':    'NIFTY',
        'BANKNIFTY':  'BANKNIFTY',
        'FINNIFTY':   'FINNIFTY',
        'MIDCPNIFTY': 'MIDCPNIFTY',
    }
    
    is_index = ticker in INDEX_MAP
    nse_symbol = INDEX_MAP.get(ticker, ticker)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com',
    }
    
    session = req.Session()
    
    try:
        # Get cookies first
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        session.get('https://www.nseindia.com/option-chain', headers=headers, timeout=10)
        
        if is_index:
            url = f'https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}'
        else:
            url = f'https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}'
        
        res = session.get(url, headers=headers, timeout=15)
        data = res.json()
        
        if 'records' not in data:
            return jsonify({'error': 'No options data from NSE'})
        
        records   = data['records']
        current_price = float(records['underlyingValue'])
        expiries  = records['expiryDates']
        expiry    = expiries[0]
        
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
        
        # Max pain
        strikes  = sorted(set([c['strike'] for c in calls_data] + [p['strike'] for p in puts_data]))
        max_pain_val = None
        min_pain = float('inf')
        for s in strikes:
            call_pain = sum(max(0, s - c['strike']) * c['openInterest'] for c in calls_data)
            put_pain  = sum(max(0, p['strike'] - s) * p['openInterest'] for p in puts_data)
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_val = s
        
        # F&O signals based on OI
        fno_signals = []
        max_call_oi = max((c['openInterest'] for c in calls_data), default=0)
        max_put_oi  = max((p['openInterest'] for p in puts_data), default=0)
        
        for c in calls_data:
            if c['openInterest'] >= max_call_oi * 0.7 and c['strike'] > current_price:
                fno_signals.append({
                    'type': 'CALL', 'strike': c['strike'],
                    'premium': c['lastPrice'], 'oi': c['openInterest'],
                    'iv': c['iv'], 'signal': 'RESISTANCE',
                    'reason': f"Highest call OI at {c['strike']} — strong resistance"
                })
        for p in puts_data:
            if p['openInterest'] >= max_put_oi * 0.7 and p['strike'] < current_price:
                fno_signals.append({
                    'type': 'PUT', 'strike': p['strike'],
                    'premium': p['lastPrice'], 'oi': p['openInterest'],
                    'iv': p['iv'], 'signal': 'SUPPORT',
                    'reason': f"Highest put OI at {p['strike']} — strong support"
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)