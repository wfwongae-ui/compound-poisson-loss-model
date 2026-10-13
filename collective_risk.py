import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pareto, poisson, norm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

plt.rcParams["font.size"] = 10
np.random.seed(42) # 固定隨機種子確保結果可復現

# ====================== 1. 模拟保单数据集 ======================
n_policies = 2000 # 总保单数量
# 协变量：risk_score (1~10，风险越高数值越大)
risk_score = np.random.uniform(1, 10, size=n_policies)
df = pd.DataFrame({"risk_score": risk_score})

# 索赔频率：泊松参数λ随risk_score上升（风险越高，索赔概率越高）
df["lambda_i"] = 0.02 + 0.015 * df["risk_score"]

# 生成每份保单索赔次数 N_i ~ Poisson(lambda_i)
df["claim_count"] = poisson.rvs(mu=df["lambda_i"], size=n_policies)

# Pareto 参数：alpha=3, theta=2000（和Example16保持一致）
alpha_pareto = 3
theta_pareto = 2000

claim_records = []
for idx, row in df.iterrows():
    n_claims = int(row["claim_count"])  # 强制转整数！修复报错
    if n_claims > 0:
        raw_losses = pareto.rvs(b=alpha_pareto, scale=theta_pareto, size=n_claims)
        for loss in raw_losses:
            claim_records.append({
                "policy_id": idx,
                "risk_score": row["risk_score"],
                "raw_X": loss
            })

claim_df = pd.DataFrame(claim_records)
print(f"总索赔记录数量: {len(claim_df)}")

# ====================== 2. 施加免赔额d、保单限额u ======================
d = 500    # ordinary deductible 普通免赔
u = 2500   # 原始损失上限policy limit
# Y = 实际赔付额
def payout(raw_x, ded, limit):
    if raw_x <= ded:
        return 0
    elif ded < raw_x <= limit:
        return raw_x - ded
    else:
        return limit - ded

claim_df["Y"] = claim_df["raw_X"].apply(lambda x: payout(x, d, u))

# 计算 E[Y], E[Y^2]（样本估计）
EY = claim_df["Y"].mean()
EY2 = (claim_df["Y"]**2).mean()
print(f"\n样本估计 E[Y] = {EY:.2f}")
print(f"样本估计 E[Y²] = {EY2:.2f}")

# ====================== 3. 线性回归：用risk_score预测severity(raw_X) ======================
X_reg = claim_df[["risk_score"]].values
y_reg = claim_df["raw_X"].values
lr = LinearRegression()
lr.fit(X_reg, y_reg)
claim_df["X_pred_lr"] = lr.predict(X_reg)
print(f"\n线性回归系数(risk_score): {lr.coef_[0]:.2f}, intercept: {lr.intercept_:.2f}")

# ====================== 4. K-means保单风险分群（原始保单，不是索赔记录） ======================
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["risk_cluster"] = kmeans.fit_predict(df[["risk_score"]])
print("\nK-means 保单风险分组统计：")
print(df["risk_cluster"].value_counts())

# ====================== 5. 复合泊松：蒙特卡洛模拟总损失S ======================
sim_times = 10000  # 模拟10000个保单组合周期
lam_total = df["lambda_i"].sum() # 整个组合总泊松λ

S_list = []
for _ in range(sim_times):
    N = int(poisson.rvs(mu=lam_total))
    if N == 0:
        S_list.append(0)
    else:
        raw_loss_sample = pareto.rvs(b=alpha_pareto, scale=theta_pareto, size=N)
        Y_sample = np.array([payout(x, d, u) for x in raw_loss_sample])
        S = np.sum(Y_sample)
        S_list.append(S)
S_arr = np.array(S_list)

# 理论 E[S], Var(S) for compound Poisson
E_S_theo = lam_total * EY
Var_S_theo = lam_total * EY2
E_S_sim = S_arr.mean()
Var_S_sim = S_arr.var(ddof=1)

print("\n==== 总损失S 理论 vs 蒙特卡洛模拟 ====")
print(f"理论 E[S] = {E_S_theo:.2f}, 模拟 E[S] = {E_S_sim:.2f}")
print(f"理论 Var(S) = {Var_S_theo:.2f}, 模拟 Var(S) = {Var_S_sim:.2f}")

# ====================== 6. CLT正态近似对比 ======================
mu_S = E_S_theo
sigma_S = np.sqrt(Var_S_theo)
x_axis = np.linspace(np.min(S_arr), np.max(S_arr), 1000)
norm_pdf = norm.pdf(x_axis, loc=mu_S, scale=sigma_S)

plt.figure(figsize=(10,5))
plt.hist(S_arr, bins=50, density=True, alpha=0.6, label="Monte Carlo Aggregate Loss S")
plt.plot(x_axis, norm_pdf, "r", linewidth=2, label="CLT Normal Approximation")
plt.xlabel("Aggregate Loss S")
plt.ylabel("Density")
plt.title("Aggregate Loss: MC Simulation vs Normal Approximation")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ====================== 7. Stop-loss Premium E[max(S - d_sl, 0)] ======================
d_sl = 8000 # stop-loss attachment point
stop_loss_prem_sim = np.mean(np.maximum(S_arr - d_sl, 0))
print(f"\nStop-loss premium (attachment={d_sl}) = {stop_loss_prem_sim:.2f}")

# ====================== 8. VaR 95% & TVaR 95% ======================
conf_level = 0.95
VaR95 = np.quantile(S_arr, conf_level)
tail_loss = S_arr[S_arr >= VaR95]
TVaR95 = tail_loss.mean()

print(f"\nVaR 95% = {VaR95:.2f}")
print(f"TVaR 95% = {TVaR95:.2f}")

plt.figure(figsize=(10,5))
plt.hist(S_arr, bins=50, density=True, alpha=0.6)
plt.axvline(VaR95, color="orange", linestyle="--", label=f"VaR95={VaR95:.0f}")
plt.axvline(TVaR95, color="red", linestyle="--", label=f"TVaR95={TVaR95:.0f}")
plt.xlabel("Aggregate Loss S")
plt.ylabel("Density")
plt.title("Aggregate Loss Distribution with VaR & TVaR")
plt.legend()
plt.show()

# ============ 新增：敏感性分析 ============
ded_list = [0, 300, 500, 800, 1000]
tvara_list = []
ES_list = []
u_fixed = 2500

for d_test in ded_list:
    def payout_test(raw_x):
        if raw_x <= d_test:
            return 0
        elif d_test < raw_x <= u_fixed:
            return raw_x - d_test
        else:
            return u_fixed - d_test

    S_test = []
    for _ in range(2000):
        N = int(poisson.rvs(mu=lam_total))
        if N ==0:
            S_test.append(0)
            continue
        xl = pareto.rvs(b=alpha_pareto, scale=theta_pareto, size=N)
        Yl = np.array([payout_test(x) for x in xl])
        S_test.append(np.sum(Yl))
    S_test_arr = np.array(S_test)
    ES_list.append(S_test_arr.mean())
    var95 = np.quantile(S_test_arr,0.95)
    tvar95 = S_test_arr[S_test_arr>=var95].mean()
    tvara_list.append(tvar95)

plt.figure(figsize=(8,4))
plt.plot(ded_list, ES_list, marker='o', label="E[S]")
plt.plot(ded_list, tvara_list, marker='s', label="TVaR95")
plt.xlabel("Deductible")
plt.ylabel("Value")
plt.title("Sensitivity: Impact of Deductible on E[S] and TVaR")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print("\n=== Sensitivity Analysis Result ===")
for d,es,tv in zip(ded_list,ES_list,tvara_list):
    print(f"Deductible={d}, E[S]={es:.2f}, TVaR95={tv:.2f}")
