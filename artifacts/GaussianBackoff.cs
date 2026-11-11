using System;

namespace MathConcepts.Retry;

/// <summary>
/// Configuration for the monotone Gaussian (CDF) retry backoff.
/// Attempt indices are 1-based: <c>k = 1</c> is the first retry after the initial failure.
/// </summary>
public sealed record BackoffOptions
{
    /// <summary>
    /// Delay before the first retry, in milliseconds.
    /// 500-900 ms is a good default range for user-facing operations.
    /// </summary>
    public int InitialMs { get; init; } = 500;

    /// <summary>
    /// Maximum delay, in milliseconds. Must be greater than <see cref="InitialMs"/>.
    /// The implementor chooses this from the acceptable worst-case wait.
    /// </summary>
    public int CapMs { get; init; } = 30_000;

    /// <summary>
    /// Ramp location in <c>n = k - 1</c> units. The exact half-cap attempt is
    /// <see cref="GaussianBackoff.HalfCapAttempt"/>; it is generally slightly
    /// greater than <c>HalfAttempts + 1</c>. Use
    /// <see cref="GaussianBackoff.HalfAttemptsForTarget"/> to solve for a target.
    /// </summary>
    public double HalfAttempts { get; init; } = 3.0;

    /// <summary>
    /// Width of the transition in attempts. <c>Sigma = RampWidthAttempts / 4</c>,
    /// so about 95% of the rise happens within
    /// <c>HalfAttempts +/- RampWidthAttempts / 2</c>.
    /// </summary>
    public double RampWidthAttempts { get; init; } = 6.0;

    /// <summary>
    /// Optional symmetric jitter fraction in <c>[0, 1)</c>.
    /// A value of 0.2 means the actual delay is uniformly between 80% and 120%
    /// of the nominal delay. Zero disables jitter.
    /// </summary>
    public double JitterFraction { get; init; } = 0.0;

    internal double Sigma => RampWidthAttempts / 4.0;

    internal void Validate()
    {
        if (InitialMs < 0)
            throw new ArgumentOutOfRangeException(nameof(InitialMs), "InitialMs must be non-negative.");
        if (CapMs <= InitialMs)
            throw new ArgumentException("CapMs must be greater than InitialMs.", nameof(CapMs));
        if (!double.IsFinite(HalfAttempts) || HalfAttempts < 0)
            throw new ArgumentOutOfRangeException(nameof(HalfAttempts), "HalfAttempts must be finite and non-negative.");
        if (!double.IsFinite(RampWidthAttempts) || RampWidthAttempts <= 0)
            throw new ArgumentOutOfRangeException(nameof(RampWidthAttempts), "RampWidthAttempts must be finite and positive.");
        if (!double.IsFinite(JitterFraction) || JitterFraction is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(JitterFraction), "JitterFraction must be finite and in [0, 1).");
    }
}

/// <summary>
/// Configuration for the optional Gaussian retry window (hump).
/// <see cref="PeakMs"/> is the height of the hump above the base schedule.
/// </summary>
public sealed record WindowOptions
{
    /// <summary>Peak contribution of the window, in milliseconds.</summary>
    public double PeakMs { get; init; } = 3_000;

    /// <summary>Attempt at which the window peaks.</summary>
    public double CenterAttempt { get; init; } = 4.0;

    /// <summary>Width of the window in attempts; roughly 95% of the hump lies within +/- 2 widths.</summary>
    public double WidthAttempts { get; init; } = 1.0;

    internal void Validate()
    {
        if (!double.IsFinite(PeakMs) || PeakMs < 0)
            throw new ArgumentOutOfRangeException(nameof(PeakMs), "PeakMs must be finite and non-negative.");
        if (!double.IsFinite(CenterAttempt) || CenterAttempt <= 0)
            throw new ArgumentOutOfRangeException(nameof(CenterAttempt), "CenterAttempt must be finite and positive.");
        if (!double.IsFinite(WidthAttempts) || WidthAttempts <= 0)
            throw new ArgumentOutOfRangeException(nameof(WidthAttempts), "WidthAttempts must be finite and positive.");
    }
}

/// <summary>Standard normal helpers used by the schedule.</summary>
public static class GaussianMath
{
    /// <summary>
    /// Error function, using the Abramowitz &amp; Stegun 7.1.26 approximation.
    /// Maximum absolute error is about 1.5e-7, which is far below the
    /// millisecond resolution of the schedule.
    /// </summary>
    public static double Erf(double x)
    {
        if (!double.IsFinite(x))
            throw new ArgumentOutOfRangeException(nameof(x), "x must be finite.");

        double sign = Math.Sign(x);
        x = Math.Abs(x);

        const double p = 0.3275911;
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.Exp(-x * x);
        return sign * y;
    }

    /// <summary>Standard normal cumulative distribution function.</summary>
    public static double Phi(double z) => 0.5 * (1.0 + Erf(z / Math.Sqrt(2.0)));

    /// <summary>Standard normal probability density function.</summary>
    public static double Pdf(double z) => Math.Exp(-0.5 * z * z) / Math.Sqrt(2.0 * Math.PI);

    /// <summary>
    /// Inverse standard normal CDF (the probit function) for <c>0 &lt; p &lt; 1</c>.
    /// Uses bisection for portability and accuracy; this is a configuration-time
    /// operation, not part of the per-retry path.
    /// </summary>
    public static double PhiInv(double p)
    {
        if (!double.IsFinite(p) || p <= 0.0 || p >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(p), "p must lie in (0, 1).");

        double lo = -40.0;
        double hi = 40.0;
        for (int i = 0; i < 100; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (Phi(mid) < p) lo = mid;
            else hi = mid;
        }

        return 0.5 * (lo + hi);
    }
}

/// <summary>Monotone, bounded retry backoff derived from the Gaussian CDF.</summary>
public static class GaussianBackoff
{
    /// <summary>
    /// Nominal (un-jittered) delay for retry attempt <paramref name="k"/>, in milliseconds.
    /// The return value is in <c>[InitialMs, CapMs]</c> and is non-decreasing in <paramref name="k"/>.
    /// </summary>
    public static double NominalDelayMs(int k, BackoffOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (k < 1)
            throw new ArgumentOutOfRangeException(nameof(k), "k is 1-based; use k = 1 for the first retry.");
        options.Validate();

        double amplitude = options.CapMs - options.InitialMs;
        double s = options.Sigma;
        double m = options.HalfAttempts;
        double n = k - 1;

        double p0 = GaussianMath.Phi(-m / s);
        double p = GaussianMath.Phi((n - m) / s);
        double fraction = (p - p0) / (1.0 - p0);

        double delay = options.InitialMs + amplitude * fraction;
        return Math.Clamp(delay, options.InitialMs, options.CapMs);
    }

    /// <summary>
    /// Integer delay for retry attempt <paramref name="k"/>, with optional jitter.
    /// Jitter is intentionally non-reproducible per call; pass a custom
    /// <see cref="Random"/> only in tests.
    /// </summary>
    public static int DelayMs(int k, BackoffOptions options, Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        double delay = NominalDelayMs(k, options);
        return RetryJitter.DelayMs(
            delay, options.JitterFraction, options.InitialMs, options.CapMs, random);
    }

    /// <summary>
    /// Exact retry attempt at which the schedule reaches half of the range,
    /// <c>k* = m + 1 + s * PhiInv((1 + Phi(-m/s)) / 2)</c>.
    /// This is the value of <c>k</c> in the continuous curve; it is generally
    /// slightly greater than <c>m + 1</c>.
    /// </summary>
    public static double HalfCapAttempt(BackoffOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        double m = options.HalfAttempts;
        double s = options.Sigma;
        double p = 0.5 * (1.0 + GaussianMath.Phi(-m / s));
        return 1.0 + m + s * GaussianMath.PhiInv(p);
    }

    /// <summary>
    /// Solves for the <c>HalfAttempts</c> value that places the exact half-cap at
    /// <paramref name="targetAttempt"/>. Uses fixed-point iteration starting from
    /// <c>m0 = targetAttempt - 1</c>, because the relationship is implicit.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The target is not reachable with a non-negative <c>m</c> for the given ramp width.
    /// </exception>
    public static double HalfAttemptsForTarget(double targetAttempt, double rampWidthAttempts)
    {
        if (targetAttempt <= 1.0)
            throw new ArgumentOutOfRangeException(nameof(targetAttempt), "Target attempt must be greater than 1.");
        if (rampWidthAttempts <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(rampWidthAttempts), "Ramp width must be positive.");

        double s = rampWidthAttempts / 4.0;
        double m = targetAttempt - 1.0;

        for (int i = 0; i < 20; i++)
        {
            double p = 0.5 * (1.0 + GaussianMath.Phi(-m / s));
            m = targetAttempt - 1.0 - s * GaussianMath.PhiInv(p);
        }

        if (m < 0.0)
            throw new ArgumentOutOfRangeException(
                nameof(targetAttempt),
                "Target is too close to 1 to be reached with a non-negative HalfAttempts for this ramp width.");

        return m;
    }
}

/// <summary>Optional Gaussian retry window (hump).</summary>
public static class GaussianWindow
{
    /// <summary>
    /// Window contribution at retry attempt <paramref name="k"/>, in milliseconds.
    /// The value is zero at the tails and <c>PeakMs</c> at <c>CenterAttempt</c>.
    /// </summary>
    public static double WindowMs(int k, WindowOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (k < 1)
            throw new ArgumentOutOfRangeException(nameof(k), "k is 1-based; use k = 1 for the first retry.");
        options.Validate();

        double z = (k - options.CenterAttempt) / options.WidthAttempts;
        return options.PeakMs * Math.Exp(-0.5 * z * z);
    }
}

/// <summary>
/// Applies optional symmetric multiplicative jitter and converts the result to
/// an integer number of milliseconds.
/// </summary>
public static class RetryJitter
{
    /// <summary>
    /// Applies symmetric multiplicative jitter to a nominal delay.
    /// Returns the nominal value unchanged when <paramref name="jitterFraction"/> is zero.
    /// </summary>
    public static double Apply(double nominalMs, double jitterFraction, Random? random = null)
    {
        if (!double.IsFinite(nominalMs))
            throw new ArgumentOutOfRangeException(nameof(nominalMs), "Nominal delay must be finite.");
        if (!double.IsFinite(jitterFraction) || jitterFraction is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(jitterFraction), "Jitter fraction must be finite and in [0, 1).");

        if (jitterFraction == 0.0)
            return nominalMs;

        random ??= Random.Shared;
        double u = random.NextDouble();
        return nominalMs * (1.0 + jitterFraction * (2.0 * u - 1.0));
    }

    /// <summary>
    /// Jitters, clamps to <c>[initialMs, capMs]</c>, and rounds to an integer.
    /// The delay is computed in <see cref="double"/> and rounded once at the end.
    /// </summary>
    public static int DelayMs(
        double nominalMs,
        double jitterFraction,
        int initialMs,
        int capMs,
        Random? random = null)
    {
        if (initialMs < 0)
            throw new ArgumentOutOfRangeException(nameof(initialMs), "Initial delay must be non-negative.");
        if (capMs < initialMs)
            throw new ArgumentOutOfRangeException(nameof(capMs), "Cap must be at least the initial delay.");

        double jittered = Apply(nominalMs, jitterFraction, random);
        double clamped = Math.Clamp(jittered, initialMs, capMs);
        return (int)Math.Round(clamped, MidpointRounding.AwayFromZero);
    }
}

/// <summary>Ways of combining a backoff curve with a retry window.</summary>
public enum WindowMode
{
    /// <summary>Add the window height to the nominal backoff delay.</summary>
    Additive,

    /// <summary>Multiply the nominal delay by <c>1 + maxMultiplier * G(k)</c>, where G peaks at 1.</summary>
    Multiplicative,

    /// <summary>Apply the multiplier to each per-attempt increment. Always monotone.</summary>
    IncrementMultiplicative
}

/// <summary>Composed schedules built from a backoff curve and an optional window.</summary>
public static class RetryComposition
{
    /// <summary>
    /// Composed nominal delay. Additive and multiplicative modes can be non-monotone;
    /// <see cref="WindowMode.IncrementMultiplicative"/> is the canonical monotone form.
    /// </summary>
    public static double NominalDelayMs(
        int k,
        BackoffOptions backoff,
        WindowOptions? window = null,
        WindowMode mode = WindowMode.Additive,
        double maxMultiplier = 1.0)
    {
        if (k < 1)
            throw new ArgumentOutOfRangeException(nameof(k));
        if (!double.IsFinite(maxMultiplier) || maxMultiplier < 0)
            throw new ArgumentOutOfRangeException(nameof(maxMultiplier), "maxMultiplier must be finite and non-negative.");
        ArgumentNullException.ThrowIfNull(backoff);

        backoff.Validate();
        double baseDelay = GaussianBackoff.NominalDelayMs(k, backoff);
        if (window is null)
            return baseDelay;

        window.Validate();
        double w = GaussianWindow.WindowMs(k, window);

        return mode switch
        {
            WindowMode.Additive =>
                Math.Clamp(baseDelay + w, backoff.InitialMs, backoff.CapMs),

            WindowMode.Multiplicative =>
                Math.Clamp(
                    baseDelay * (1.0 + maxMultiplier * NormalizedWindow(k, window)),
                    backoff.InitialMs,
                    backoff.CapMs),

            WindowMode.IncrementMultiplicative =>
                IncrementWindowDelay(k, backoff, window, maxMultiplier),

            _ => throw new ArgumentOutOfRangeException(nameof(mode))
        };
    }

    private static double NormalizedWindow(int k, WindowOptions window)
        => window.PeakMs <= 0 ? 0.0 : GaussianWindow.WindowMs(k, window) / window.PeakMs;

    private static double IncrementWindowDelay(
        int k,
        BackoffOptions backoff,
        WindowOptions window,
        double maxMultiplier)
    {
        double delay = backoff.InitialMs;
        double previousNominal = backoff.InitialMs;

        for (int j = 1; j <= k; j++)
        {
            double nominal = GaussianBackoff.NominalDelayMs(j, backoff);
            double increment = nominal - previousNominal;
            double multiplier = 1.0 + maxMultiplier * NormalizedWindow(j, window);
            delay += increment * multiplier;
            previousNominal = nominal;
        }

        return Math.Clamp(delay, backoff.InitialMs, backoff.CapMs);
    }
}

/// <summary>Cumulative schedule for the capped increment-multiplicative composition.</summary>
public sealed class PrecomputedSchedule
{
    private readonly double[] _delays; // index 0 unused; k = 1..capacity

    /// <summary>
    /// Precomputes the nominal schedule once. Use this when many operations share
    /// the same configuration; invalidate the instance when the configuration changes.
    /// </summary>
    public PrecomputedSchedule(
        BackoffOptions backoff,
        WindowOptions? window = null,
        double maxMultiplier = 1.0,
        int capacity = 32)
    {
        ArgumentNullException.ThrowIfNull(backoff);
        backoff.Validate();
        if (!double.IsFinite(maxMultiplier) || maxMultiplier < 0)
            throw new ArgumentOutOfRangeException(nameof(maxMultiplier), "maxMultiplier must be finite and non-negative.");
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be at least 1.");

        window?.Validate();
        _delays = new double[capacity + 1];

        double delay = backoff.InitialMs;
        double previousNominal = backoff.InitialMs;
        _delays[1] = delay;

        for (int k = 2; k <= capacity; k++)
        {
            double nominal = GaussianBackoff.NominalDelayMs(k, backoff);
            double increment = nominal - previousNominal;
            double g = window is null
                ? 0.0
                : (window.PeakMs <= 0 ? 0.0 : GaussianWindow.WindowMs(k, window) / window.PeakMs);

            delay += increment * (1.0 + maxMultiplier * g);
            _delays[k] = Math.Min(backoff.CapMs, delay);
            previousNominal = nominal;
        }
    }

    /// <summary>Precomputed nominal delay for retry attempt <paramref name="k"/>.</summary>
    public double NominalDelayMs(int k)
    {
        if (k < 1 || k >= _delays.Length)
            throw new ArgumentOutOfRangeException(nameof(k), "k is outside the precomputed range.");
        return _delays[k];
    }
}

/// <summary>
/// Backwards-compatible facade over <see cref="RetryComposition"/>.
/// New code should use the layered types directly.
/// </summary>
public static class RetrySchedule
{
    /// <inheritdoc cref="RetryComposition.NominalDelayMs"/>
    public static double NominalDelayMs(
        int k,
        BackoffOptions backoff,
        WindowOptions? window = null,
        WindowMode mode = WindowMode.Additive,
        double maxMultiplier = 1.0)
        => RetryComposition.NominalDelayMs(k, backoff, window, mode, maxMultiplier);
}

/// <summary>
/// Converts a local scheduling delay into the delay actually awaited, honoring
/// server-directed timing. The local cap does not bound the effective delay.
/// </summary>
public static class RetryPolicy
{
    /// <summary>
    /// Returns <c>max(localDelayMs, retryAfterMs)</c>. A <c>Retry-After</c> value
    /// may exceed the local scheduling cap; that is intentional.
    /// </summary>
    public static int EffectiveDelayMs(int localDelayMs, int? retryAfterMs = null)
    {
        if (localDelayMs < 0)
            throw new ArgumentOutOfRangeException(nameof(localDelayMs), "Local delay must be non-negative.");
        if (retryAfterMs is < 0)
            throw new ArgumentOutOfRangeException(nameof(retryAfterMs), "Retry-After must be non-negative.");

        return retryAfterMs is null ? localDelayMs : Math.Max(localDelayMs, retryAfterMs.Value);
    }
}

/// <summary>Small async retry helper that uses the Gaussian backoff.</summary>
public static class RetryExecutor
{
    /// <summary>
    /// Executes <paramref name="operation"/> until it succeeds, it is cancelled,
    /// <paramref name="shouldRetry"/> rejects the exception, or
    /// <paramref name="maxAttempts"/> total attempts have been made.
    /// </summary>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="options">Backoff configuration.</param>
    /// <param name="maxAttempts">Total attempts, including the initial one. Must be at least 1.</param>
    /// <param name="shouldRetry">Optional predicate; return false for non-transient failures.</param>
    /// <param name="onRetry">Optional callback invoked with (retryIndex, delayMs) before sleeping.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <param name="retryAfterMs">
    /// Optional server-directed delay for the failed attempt, in milliseconds. The effective
    /// delay is <c>max(localDelay, retryAfterMs)</c> and may exceed the local cap.
    /// </param>
    public static async Task<T> ExecuteAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        BackoffOptions options,
        int maxAttempts,
        Func<Exception, bool>? shouldRetry = null,
        Action<int, int>? onRetry = null,
        CancellationToken cancellationToken = default,
        Func<Exception, int?>? retryAfterMs = null)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (maxAttempts < 1)
            throw new ArgumentOutOfRangeException(nameof(maxAttempts), "maxAttempts must be at least 1.");

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            try
            {
                return await operation(cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex) when (attempt < maxAttempts - 1 && (shouldRetry?.Invoke(ex) ?? true))
            {
                int retryIndex = attempt + 1; // k = 1 is the first retry after the initial failure
                int localDelay = GaussianBackoff.DelayMs(retryIndex, options);
                int delay = RetryPolicy.EffectiveDelayMs(localDelay, retryAfterMs?.Invoke(ex));
                onRetry?.Invoke(retryIndex, delay);
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }

        // Unreachable in normal operation: every path returns, rethrows, or exhausts
        // its attempts. Present to satisfy the compiler's flow analysis.
        throw new InvalidOperationException("Retry loop exited without a result or exception.");
    }
}
