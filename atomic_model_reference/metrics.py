"""Exact integrals of declared piecewise-linear curves, without extrapolation."""

import bisect
import math


def interpolate(times, values, t):
    if t < times[0] or t > times[-1]:
        raise ValueError("Interpolation would extrapolate")
    i = bisect.bisect_right(times, t) - 1
    if i == len(times) - 1:
        return values[-1]
    f = (t - times[i]) / (times[i + 1] - times[i])
    return values[i] + f * (values[i + 1] - values[i])


def grid(times, start, end):
    if start < times[0] or end > times[-1]:
        raise ValueError("Curve does not cover comparison window")
    return sorted({start, end, *(t for t in times if start < t < end)})


def max_gap(times, start, end):
    # Restrict the check to intervals actually used for interpolation.
    return max((b - a for a, b in zip(times, times[1:]) if b > start and a < end), default=0.0)


def first_crossing(times, values, start, end, threshold, direction):
    points = grid(times, start, end)
    for a, b in zip(points, points[1:]):
        x = interpolate(times, values, a)
        y = interpolate(times, values, b)
        crossed = x < threshold <= y if direction == "up" else x > threshold >= y
        if crossed:
            return a + (b - a) * (threshold - x) / (y - x)
    return None


def _covered_fraction(a, b, s0, s1):
    roots = {0.0, 1.0}
    for x, y in ((a - s0, b - s1), (-a - s0, -b - s1)):
        if x != y:
            root = -x / (y - x)
            if 0 < root < 1:
                roots.add(root)
    points = sorted(roots)
    covered = 0.0
    for lo, hi in zip(points, points[1:]):
        f = (lo + hi) / 2
        if abs(a + f * (b - a)) <= s0 + f * (s1 - s0):
            covered += hi - lo
    return covered


def compare_curve(model_t, model_y, ref_t, ref_y, window, uncertainty=None):
    start, end = window
    duration = end - start
    points = sorted(set(grid(model_t, start, end)) | set(grid(ref_t, start, end)))
    model = [interpolate(model_t, model_y, t) for t in points]
    reference = [interpolate(ref_t, ref_y, t) for t in points]
    error = [a - b for a, b in zip(model, reference)]
    signed = []
    absolute = []
    squared = []
    mi = []
    ri = []
    cover = []
    sigma = None if uncertainty is None else [interpolate(ref_t, uncertainty, t) for t in points]
    for i, (a, b) in enumerate(zip(points, points[1:])):
        h = b - a
        x, y = error[i : i + 2]
        signed.append(h * (x + y) / 2)
        if x < 0 < y or y < 0 < x:
            area = h * (x * x + y * y) / (2 * (abs(x) + abs(y)))
        else:
            area = h * (abs(x) + abs(y)) / 2
        absolute.append(area)
        squared.append(h * (x * x + x * y + y * y) / 3)
        mi.append(h * (model[i] + model[i + 1]) / 2)
        ri.append(h * (reference[i] + reference[i + 1]) / 2)
        if sigma is not None:
            cover.append(h * _covered_fraction(x, y, sigma[i], sigma[i + 1]))
    mpeak = max(range(len(model)), key=lambda i: model[i])
    rpeak = max(range(len(reference)), key=lambda i: reference[i])
    return dict(
        duration_s=duration,
        comparison_knots=len(points),
        bias=math.fsum(signed) / duration,
        mae=math.fsum(absolute) / duration,
        rmse=math.sqrt(math.fsum(squared) / duration),
        peak_absolute_error=max(map(abs, error)),
        integral_absolute_error=math.fsum(absolute),
        integral_error=math.fsum(signed),
        model_integral=math.fsum(mi),
        reference_integral=math.fsum(ri),
        model_peak=model[mpeak],
        reference_peak=reference[rpeak],
        model_peak_time_s=points[mpeak],
        reference_peak_time_s=points[rpeak],
        peak_value_error=model[mpeak] - reference[rpeak],
        peak_time_error_s=points[mpeak] - points[rpeak],
        fraction_time_within_declared_uncertainty=math.fsum(cover) / duration
        if sigma is not None
        else None,
    )
