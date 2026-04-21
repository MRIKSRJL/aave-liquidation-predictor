import pandas as pd
import json
import requests
import time

FILE_HEALTHY = "aave_raw_users.jsonl"
FILE_LIQUIDATED = "aave_liquidated_users.jsonl"
OUTPUT_FILE = "aave_dataset_advanced.csv"

# Token symbol mapping (Aave symbol -> CoinGecko ID)
TOKEN_MAP = {
    'WBTC': 'bitcoin', 'WETH': 'ethereum', 'WMATIC': 'matic-network',
    'USDC': 'usd-coin', 'USDC.e': 'usd-coin', 'USDT': 'tether',
    'DAI': 'dai', 'AAVE': 'aave', 'LINK': 'chainlink',
    'CRV': 'curve-dao-token', 'BAL': 'balancer'
}
COINGECKO_TIMEOUT_SECONDS = 10
MAX_PRICE_RETRIES = 3
STABLE_SYMBOLS = {"USDC", "USDC.e", "USDT", "DAI"}
VOLATILE_SYMBOLS = {"WBTC", "WETH", "WMATIC", "AAVE", "LINK", "CRV", "BAL"}

def fetch_live_prices():
    print("Connecting to CoinGecko API to retrieve live token prices...")
    coin_ids = ",".join(set(TOKEN_MAP.values()))
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_ids}&vs_currencies=usd"

    for attempt in range(1, MAX_PRICE_RETRIES + 1):
        try:
            response = requests.get(url, timeout=COINGECKO_TIMEOUT_SECONDS)
            response.raise_for_status()
            print("Live prices fetched successfully.")
            return response.json()
        except requests.RequestException as error:
            print(f"[Retry {attempt}/{MAX_PRICE_RETRIES}] Price API error: {error}")
            if attempt < MAX_PRICE_RETRIES:
                time.sleep(attempt)
        except ValueError as error:
            print(f"[Retry {attempt}/{MAX_PRICE_RETRIES}] Invalid price payload: {error}")
            if attempt < MAX_PRICE_RETRIES:
                time.sleep(attempt)

    print("Falling back to static reference prices after repeated API failures.")
    return {
        'bitcoin': {'usd': 65000}, 'ethereum': {'usd': 3500},
        'matic-network': {'usd': 0.70}, 'usd-coin': {'usd': 1},
        'tether': {'usd': 1}, 'dai': {'usd': 1}, 'aave': {'usd': 100}
    }

def parse_positions_in_usd(positions, live_prices):
    total_collateral_usd = 0
    total_debt_usd = 0
    collateral_by_symbol = {}
    debt_by_symbol = {}
    num_lend_positions = 0
    num_borrow_positions = 0

    for pos in positions:
        balance = float(pos.get('balance', 0))
        token_data = pos.get('market', {}).get('inputToken', {})
        decimals = int(token_data.get('decimals', 18))
        symbol = token_data.get('symbol', 'UNKNOWN')
        side = pos.get('side')
        
        adjusted_balance = balance / (10 ** decimals)
        
        cg_id = TOKEN_MAP.get(symbol)
        price_usd = live_prices.get(cg_id, {}).get('usd', 0)
        value_usd = adjusted_balance * price_usd

        if side == 'LENDER':
            total_collateral_usd += value_usd
            num_lend_positions += 1
            collateral_by_symbol[symbol] = collateral_by_symbol.get(symbol, 0) + value_usd
        elif side == 'BORROWER':
            total_debt_usd += value_usd
            num_borrow_positions += 1
            debt_by_symbol[symbol] = debt_by_symbol.get(symbol, 0) + value_usd

    debt_stable_usd = sum(
        value for symbol, value in debt_by_symbol.items() if symbol in STABLE_SYMBOLS
    )
    collateral_volatile_usd = sum(
        value for symbol, value in collateral_by_symbol.items() if symbol in VOLATILE_SYMBOLS
    )

    debt_stable_share = debt_stable_usd / total_debt_usd if total_debt_usd > 0 else 0
    collateral_volatile_share = (
        collateral_volatile_usd / total_collateral_usd if total_collateral_usd > 0 else 0
    )

    debt_concentration_hhi = (
        sum((value / total_debt_usd) ** 2 for value in debt_by_symbol.values())
        if total_debt_usd > 0
        else 0
    )
    collateral_concentration_hhi = (
        sum((value / total_collateral_usd) ** 2 for value in collateral_by_symbol.values())
        if total_collateral_usd > 0
        else 0
    )

    return {
        "total_collateral_usd": total_collateral_usd,
        "total_debt_usd": total_debt_usd,
        "num_lend_positions": num_lend_positions,
        "num_borrow_positions": num_borrow_positions,
        "num_debt_assets": len(debt_by_symbol),
        "debt_stable_share": debt_stable_share,
        "collateral_volatile_share": collateral_volatile_share,
        "debt_concentration_hhi": debt_concentration_hhi,
        "collateral_concentration_hhi": collateral_concentration_hhi,
    }

def load_advanced_features(input_path, live_prices, is_liquidated_source=False):
    print(f"Processing source file: {input_path}")
    records = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    acc = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Skipping malformed JSON line {line_number} in {input_path}."
                    )
                    continue
                metrics = parse_positions_in_usd(acc.get('positions', []), live_prices)
                collat_usd = metrics["total_collateral_usd"]
                debt_usd = metrics["total_debt_usd"]

                if collat_usd == 0 and debt_usd == 0:
                    continue
                if not is_liquidated_source and debt_usd == 0:
                    continue

                ltv = debt_usd / collat_usd if collat_usd > 0 else 0

                records.append({
                    'account_id': acc['id'],
                    'total_collateral_usd': collat_usd,
                    'total_debt_usd': debt_usd,
                    'ltv': ltv,
                    'num_positions': len(acc.get('positions', [])),
                    'num_lend_positions': metrics["num_lend_positions"],
                    'num_borrow_positions': metrics["num_borrow_positions"],
                    'num_debt_assets': metrics["num_debt_assets"],
                    'debt_stable_share': metrics["debt_stable_share"],
                    'collateral_volatile_share': metrics["collateral_volatile_share"],
                    'debt_concentration_hhi': metrics["debt_concentration_hhi"],
                    'collateral_concentration_hhi': metrics["collateral_concentration_hhi"],
                    'is_liquidated': 1 if (is_liquidated_source or len(acc.get('liquidates', [])) > 0) else 0
                })
    except FileNotFoundError:
        print(f"Source file not found: {input_path}. Continuing with an empty dataset.")
    return pd.DataFrame(records)

if __name__ == "__main__":
    live_prices = fetch_live_prices()
    df_h = load_advanced_features(FILE_HEALTHY, live_prices, False)
    df_l = load_advanced_features(FILE_LIQUIDATED, live_prices, True)
    
    df_final = pd.concat([df_h, df_l]).drop_duplicates(subset='account_id', keep='first')
    df_final = df_final[df_final['ltv'] < 100] 
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinancial feature dataset generated: {OUTPUT_FILE}")