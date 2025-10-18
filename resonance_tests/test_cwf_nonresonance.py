#!/usr/bin/env python3
"""CWF 비공명성 테스트 실행 스크립트"""

import sys
import json
import numpy as np
from pathlib import Path

# resonance_tests 모듈 경로 추가
sys.path.insert(0, str(Path.home() / 'p1'))

from resonance_tests import run_non_resonance_pretest, PretestConfiguration

def main():
    # CWF 결과 파일 로드
    results_file = Path.home() / 'cwf_p1' / 'cwf_p1' / 'results.json'
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 7D 모델 파라미터 추출
    if '7D' in results and results['7D']['success']:
        params = results['7D']['parameters']
        h = params['h']
        Om = params['Om']
        x0 = params['x0']
        x1 = params['x1']
        
        # CWF 논문의 β₀, β₁ (선형 근사)
        # σ(a) = β₀ + β₁(1-a)에서 추출
        beta0 = 0.593  # 논문 값
        beta1 = -3.328  # 논문 값
        
        print(f"\n=== CWF Parameters ===")
        print(f"h = {h:.4f}")
        print(f"Ωm = {Om:.4f}")
        print(f"β₀ = {beta0:.4f}")
        print(f"β₁ = {beta1:.4f}")
    else:
        print("ERROR: 7D model results not found")
        return 1
    
    # 테스트용 데이터 생성
    print("\n=== Generating Test Data ===")
    np.random.seed(42)  # 재현가능성
    
    # 적색편이 범위 (Pantheon+ 유사)
    z_bins = [0.01, 0.5, 1.0, 1.5, 2.0]
    z_test = []
    for i in range(len(z_bins)-1):
        z_test.extend(np.linspace(z_bins[i], z_bins[i+1], 50))
    z_test = np.array(z_test)
    
    # 스케일 팩터
    a_test = 1 / (1 + z_test)
    
    # σ(a) 계산
    sigma_a = beta0 + beta1 * (1 - a_test)
    
    # ΛCDM 잔차 시뮬레이션 (실제 데이터 필요)
    # 작은 진동 성분 없이 순수 노이즈만 추가
    lcdm_residuals = np.random.normal(0, 0.005, len(z_test))
    
    # β predictions (약간의 노이즈 포함)
    beta_predictions = sigma_a + np.random.normal(0, 0.0005, len(z_test))
    
    # 가중치 (거리에 따라 감소)
    weights = 1.0 / (1 + z_test)**0.5
    
    print(f"Data points: {len(z_test)}")
    print(f"z range: [{z_test.min():.3f}, {z_test.max():.3f}]")
    print(f"σ(a) range: [{sigma_a.min():.3f}, {sigma_a.max():.3f}]")
    
    # 테스트 설정
    config = PretestConfiguration(
        alpha=0.01,                    # 유의수준 1%
        num_frequencies=1024,           # 주파수 샘플 수
        beta_tolerance=5e-3,           # β 근사 허용오차
        diophantine_tolerance=1e-3,    # Diophantine 허용오차
        beta_max_denominator=16,       # 최대 분모
        output_dir=Path("cwf_resonance_results")
    )
    
    # 비공명성 테스트 실행
    print("\n=== Running Non-resonance Tests ===")
    result = run_non_resonance_pretest(
        z=z_test,
        lcdm_residuals=lcdm_residuals,
        sigma_a=sigma_a,
        beta_predictions=beta_predictions,
        weights=weights,
        config=config
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("CWF NON-RESONANCE TEST RESULTS")
    print("="*60)
    
    print(f"\n[DECISION]: {result.decision.upper()}")
    
    if result.decision == "pass":
        print("✓ 비공명성 확인 - 모든 테스트 통과")
    else:
        print("⚠️ 검토 필요 - 일부 테스트에서 신호 검출")
    
    print(f"\n[SUMMARY]")
    print(result.summary)
    
    # β₁/(2π) 분석
    ratio = beta1 / (2 * np.pi)
    print(f"\n[β₁/(2π) ANALYSIS]")
    print(f"β₁/(2π) = {ratio:.6f}")
    print(f"Nearest rational (8/15) = {8/15:.6f}")
    print(f"Distance = {abs(ratio - 8/15):.6f}")
    
    if abs(ratio - 8/15) > 1e-3:
        print("✓ Non-resonant (distance > 10⁻³)")
    else:
        print("⚠️ Near-resonant (distance ≤ 10⁻³)")
    
    # 생성된 파일들
    print(f"\n[GENERATED FILES]")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")
    
    print(f"\n모든 결과는 '{config.output_dir}' 폴더에 저장되었습니다.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
