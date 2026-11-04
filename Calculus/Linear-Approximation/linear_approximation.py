
"""ApproxLab V2.2: Linear Approximation Calculator.

Features:
1. User-defined mathematical functions
2. Automatic differentiation
3. Linear approximation
4. Absolute and relative error
5. Success/Failed determination
6. Automatic estimation of valid approximation interval
7. PNG visualization

Requirements: sympy, numpy, matplotlib
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


x = sp.Symbol('x', real=True)
SAMPLES_PER_SIDE = 1600


def real_number(expression, label):
    """Convert a symbolic expression to a finite real number."""
    try:
        result = float(sp.N(expression))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{label} must be a finite real number.'
        )

    if not math.isfinite(result):
        raise ValueError(
            f'{label} must be a finite real number.'
        )

    return result


def safe_eval(func, value):
    """Evaluate a function safely within its real domain."""
    try:
        with np.errstate(all='ignore'):
            result = func(float(value))

        result = complex(result)

        if abs(result.imag) > 1e-10:
            return None

        if not math.isfinite(result.real):
            return None

        return result.real

    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None


def accuracy_percent(f_func, approx_func, value):
    """Calculate relative error as a percentage."""

    actual = safe_eval(f_func, value)
    approx = safe_eval(approx_func, value)

    if actual is None or approx is None or actual == 0:
        return None

    return abs(approx - actual) / abs(actual) * 100.0


def find_boundary(
    f_func,
    approx_func,
    a,
    tolerance,
    radius,
    direction
):
    """Estimate one boundary of the valid interval."""

    previous_good = a

    for step in range(1, SAMPLES_PER_SIDE + 1):

        trial = (
            a
            + direction * radius * step / SAMPLES_PER_SIDE
        )

        error = accuracy_percent(
            f_func,
            approx_func,
            trial
        )

        if error is not None and error <= tolerance:
            previous_good = trial
            continue

        # Bisection method
        good = previous_good
        bad = trial

        for _ in range(55):

            midpoint = (good + bad) / 2.0

            if midpoint == good or midpoint == bad:
                break

            midpoint_error = accuracy_percent(
                f_func,
                approx_func,
                midpoint
            )

            if (
                midpoint_error is not None
                and midpoint_error <= tolerance
            ):
                good = midpoint
            else:
                bad = midpoint

        return good, False

    return a + direction * radius, True


def approximate_valid_interval(
    f_func,
    approx_func,
    a,
    tolerance,
    radius
):
    """Estimate the valid interval around a."""

    if accuracy_percent(f_func, approx_func, a) is None:
        return None

    left, left_limited = find_boundary(
        f_func,
        approx_func,
        a,
        tolerance,
        radius,
        -1
    )

    right, right_limited = find_boundary(
        f_func,
        approx_func,
        a,
        tolerance,
        radius,
        +1
    )

    return left, right, left_limited, right_limited


def values_for_plot(func, xs):
    """Evaluate valid function values for plotting."""

    return np.array([
        (v if v is not None else np.nan)
        for v in (
            safe_eval(func, position)
            for position in xs
        )
    ])


def create_plot(
    f_expr,
    L_expr,
    a,
    target,
    interval,
    radius,
    filename
):
    """Generate and save the visualization."""

    f_func = sp.lambdify(
        x, f_expr, modules='numpy'
    )

    L_func = sp.lambdify(
        x, L_expr, modules='numpy'
    )

    xs = np.linspace(
        a - radius,
        a + radius,
        1000
    )

    yf = values_for_plot(f_func, xs)
    yl = values_for_plot(L_func, xs)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Highlight valid approximation interval
    if interval is not None:

        left, right, left_limited, right_limited = interval

        ax.axvspan(
            left,
            right,
            color='#34a853',
            alpha=0.16,
            label='Estimated valid interval'
        )

        if not left_limited:
            ax.axvline(
                left,
                color='#258642',
                linestyle=':',
                alpha=0.8
            )

        if not right_limited:
            ax.axvline(
                right,
                color='#258642',
                linestyle=':',
                alpha=0.8
            )

    # Original function
    ax.plot(
        xs,
        yf,
        color='#1967d2',
        linewidth=2.3,
        label='Original f(x)'
    )

    # Linear approximation
    ax.plot(
        xs,
        yl,
        color='#ea8600',
        linewidth=2.1,
        linestyle='--',
        label='Linear approximation L(x)'
    )

    # Important points
    fa = safe_eval(f_func, a)
    ft = safe_eval(f_func, target)
    lt = safe_eval(L_func, target)

    if fa is not None:
        ax.scatter(
            [a],
            [fa],
            color='#242424',
            s=55,
            zorder=5,
            label='Expansion point a'
        )

    if ft is not None:
        ax.scatter(
            [target],
            [ft],
            color='#1967d2',
            s=70,
            zorder=5,
            label='Actual value at target'
        )

    if lt is not None:
        ax.scatter(
            [target],
            [lt],
            color='#ea8600',
            s=55,
            zorder=5,
            label='Approximation at target'
        )

    ax.axvline(
        a,
        color='#777777',
        linewidth=1,
        linestyle=':',
        alpha=0.6
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_title(
        'Linear Approximation | Estimated Accuracy Interval'
    )

    ax.grid(alpha=0.22)
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main():

    print(
        '=== ApproxLab V2.2: '
        'Linear Approximation + Valid Interval ==='
    )

    print(
        'Functions: x**2, sqrt(x), sin(x), '
        'exp(x), log(x), Abs(x)'
    )

    try:
        # User inputs
        raw_function = input('Enter f(x): ').strip()

        if not raw_function:
            raise ValueError(
                'Please enter a function (example: x**2).'
            )

        f = sp.sympify(
            raw_function,
            locals={'x': x}
        )

        if not isinstance(f, sp.Expr) or f.free_symbols - {x}:
            raise ValueError(
                'Use a mathematical expression with only x.'
            )

        a_sym = sp.nsimplify(
            sp.sympify(
                input('Enter expansion point a: ')
            )
        )

        target_sym = sp.nsimplify(
            sp.sympify(
                input('Enter target x: ')
            )
        )

        tolerance = float(
            input('Enter allowed relative error (%): ')
        )

        file_name = input(
            'Image name (Enter for test.png): '
        ).strip() or 'test.png'

        # Ensure PNG extension
        file_path = Path(file_name).with_suffix('.png')

        a = real_number(
            a_sym,
            'Expansion point'
        )

        target = real_number(
            target_sym,
            'Target point'
        )

        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                'Allowed relative error must be >= 0 and finite.'
            )

        # Calculate f(a)
        f_at_a = f.subs(x, a_sym)
        real_number(f_at_a, 'f(a)')

        # Check differentiability
        difference_quotient = (
            f - f_at_a
        ) / (x - a_sym)

        left_derivative = sp.limit(
            difference_quotient,
            x,
            a_sym,
            dir='-'
        )

        right_derivative = sp.limit(
            difference_quotient,
            x,
            a_sym,
            dir='+'
        )

        dl = real_number(
            left_derivative,
            'Left derivative'
        )

        dr = real_number(
            right_derivative,
            'Right derivative'
        )

        if not math.isclose(
            dl,
            dr,
            rel_tol=1e-9,
            abs_tol=1e-9
        ):
            print(
                'Cannot linearize: '
                'the function is not differentiable at a.'
            )
            return

        # Calculate linear approximation
        L = sp.simplify(
            f_at_a
            + left_derivative * (x - a_sym)
        )

        f_func = sp.lambdify(
            x, f, modules='numpy'
        )

        L_func = sp.lambdify(
            x, L, modules='numpy'
        )

        # Calculate values
        approximation = safe_eval(
            L_func, target
        )

        actual = safe_eval(
            f_func, target
        )

        if approximation is None or actual is None:
            raise ValueError(
                'The function is not real/finite '
                'at the target point.'
            )

        # Calculate relative error
        relative_error = accuracy_percent(
            f_func,
            L_func,
            target
        )

        # Determine search range
        radius = max(
            2.0,
            2.0 * abs(target - a),
            0.5 * abs(a)
        )

        # NEW: Find valid interval
        interval = approximate_valid_interval(
            f_func,
            L_func,
            a,
            tolerance,
            radius
        )

        # Display results
        print('\n--- Results ---')

        print(f'f(x) = {f}')
        print(f"f'(x) = {sp.diff(f, x)}")
        print(f'L(x) = {L}')

        print(
            f'Approximation = {approximation:.6f}'
        )

        print(
            f'Actual value = {actual:.6f}'
        )

        print(
            f'Absolute error = '
            f'{abs(approximation - actual):.6f}'
        )

        if relative_error is None:

            print(
                'Relative error = undefined '
                '(actual value is 0)'
            )

            print(
                'No Success/Failed verdict '
                'for this target point.'
            )

        else:

            print(
                f'Relative error = {relative_error:.6f}%'
            )

            if relative_error <= tolerance:
                print('Success!')
            else:
                print('Failed!')

        # Display valid interval
        print(
            '\n--- NEW: Estimated valid interval ---'
        )

        if interval is None:

            print(
                'No interval containing a: '
                'relative error is undefined at a.'
            )

            print(
                'Try a point with f(a) != 0, '
                'or use absolute error instead.'
            )

        else:

            left, right, left_limited, right_limited = interval

            print(
                f'Estimated interval near a: '
                f'[{left:.6f}, {right:.6f}]'
            )

            print(
                f'Condition: relative error <= {tolerance:g}%'
            )

            if left_limited or right_limited:
                print(
                    'At least one endpoint reached '
                    'the chosen search boundary.'
                )

                print(
                    'The actual valid range may extend farther.'
                )

            print(
                'Note: sampled + bisection estimate; '
                'NOT a rigorous proof.'
            )

        # Generate image
        create_plot(
            f,
            L,
            a,
            target,
            interval,
            radius,
            file_path
        )

        print(
            f'PNG saved: {file_path.resolve()}'
        )

    except (
        sp.SympifyError,
        ValueError,
        TypeError,
        ArithmeticError,
        NotImplementedError,
        AttributeError,
        OSError
    ) as error:

        print(f'Error: {error}')


if __name__ == '__main__':
    main()
