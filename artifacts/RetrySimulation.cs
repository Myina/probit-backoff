using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using MathConcepts.Retry;

// Reproducible synthetic comparison of retry schedules under a synchronized
// outage followed by partial recovery. This is an illustrative simulation,
// not a measurement of any real service.
//
// To run: create a console project, replace its Program.cs with this file,
// add GaussianBackoff.cs to the project, and run `dotnet run`.
// The program writes retry-simulation-results.csv in the working directory
// and prints a summary table.

const int ClientCount = 2_000;
const int MaxAttempts = 8;
const int HorizonMs = 180_000;
const int BucketMs = 100;
const double BaseMs = 500.0;
const double LocalCapMs = 30_000.0;

var probit = new BackoffOptions
{
    InitialMs = 500,
    CapMs = 30_000,
    HalfAttempts = 3.0,
    RampWidthAttempts = 6.0,
    JitterFraction = 0.0
};

var strategies = new (string Name, Func<RetryContext, double> Delay)[]
{
    ("exponential", c => Exponential(c.K)),
    ("exponential-full-jitter", c => Uniform(c, 1) * Exponential(c.K)),
    ("exponential-equal-jitter", c =>
    {
        double d = Exponential(c.K);
        return 0.5 * d + Uniform(c, 1) * 0.5 * d;
    }),
    ("exponential-decorrelated-jitter", c =>
        Math.Min(LocalCapMs, BaseMs + Uniform(c, 1) * (Math.Max(BaseMs, c.PreviousDelay * 3.0) - BaseMs))),
    ("probit", c => GaussianBackoff.NominalDelayMs(c.K, probit)),
    ("probit-symmetric-jitter", c =>
    {
        double d = GaussianBackoff.NominalDelayMs(c.K, probit);
        return d * (1.0 + 0.2 * (2.0 * Uniform(c, 1) - 1.0));
    })
};

var results = new List<SimulationResult>();
foreach (var strategy in strategies)
{
    results.Add(Run(strategy.Name, strategy.Delay));
}

const string csvPath = "retry-simulation-results.csv";
using (var writer = new StreamWriter(csvPath))
{
    writer.WriteLine("strategy,successRate,p50Ms,p95Ms,p99Ms,peakRetryArrivalsPer100ms,totalAttempts,meanAttempts,peakAtMs");
    foreach (var r in results)
    {
        writer.WriteLine(string.Join(',',
            r.Name,
            r.SuccessRate.ToString("F4", CultureInfo.InvariantCulture),
            r.P50.ToString(CultureInfo.InvariantCulture),
            r.P95.ToString(CultureInfo.InvariantCulture),
            r.P99.ToString(CultureInfo.InvariantCulture),
            r.PeakRetryArrivals,
            r.TotalAttempts,
            r.MeanAttempts.ToString("F2", CultureInfo.InvariantCulture),
            r.PeakAtMs));
    }
}

Console.WriteLine($"clients={ClientCount} maxAttempts={MaxAttempts} horizon={HorizonMs}ms");
Console.WriteLine("strategy                         success  p50     p95     p99     peakRetry/100ms  attempts  peakAt");
foreach (var r in results)
{
    Console.WriteLine(
        $"{r.Name,-32} {r.SuccessRate,7:P1} {r.P50,7} {r.P95,7} {r.P99,7} {r.PeakRetryArrivals,16} {r.TotalAttempts,9} {r.PeakAtMs,8}");
}
Console.WriteLine($"wrote {Path.GetFullPath(csvPath)}");

static double Exponential(int k)
    => Math.Min(LocalCapMs, BaseMs * Math.Pow(2.0, k - 1));

static double Health(int timeMs)
{
    if (timeMs < 2_000) return 0.0;
    if (timeMs < 12_000) return (timeMs - 2_000) / 10_000.0;
    return 1.0;
}

static double Uniform(RetryContext c, int stream)
    => Uniform01(c.Client * 1_000_003L + c.Attempt * 7_919L + stream * 104_729L);

static double Uniform01(long seed)
{
    ulong z = unchecked((ulong)seed + 0x9E3779B97F4A7C15UL);
    z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
    z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
    z ^= z >> 31;
    return (z >> 11) * (1.0 / (1UL << 53));
}

static SimulationResult Run(string name, Func<RetryContext, double> delay)
{
    var queue = new PriorityQueue<(int Client, int Attempt, double PreviousDelay), int>();
    var completions = new List<int>(ClientCount);
    var retryArrivals = new int[HorizonMs / BucketMs + 1];
    int failed = 0;
    long totalAttempts = 0;

    for (int client = 0; client < ClientCount; client++)
        queue.Enqueue((client, 1, BaseMs), 0);

    while (queue.TryDequeue(out var item, out int time))
    {
        if (time > HorizonMs)
        {
            failed++;
            continue;
        }

        var context = new RetryContext(item.Client, item.Attempt, item.PreviousDelay);
        if (item.Attempt > 1)
            retryArrivals[time / BucketMs]++;
        totalAttempts++;

        bool success = Uniform01(
            context.Client * 1_000_003L + context.Attempt * 7_919L) < Health(time);

        if (success)
        {
            completions.Add(time);
            continue;
        }

        if (item.Attempt >= MaxAttempts)
        {
            failed++;
            continue;
        }

        double nextDelay = Math.Max(1.0, Math.Ceiling(delay(context)));
        int nextTime = time + (int)nextDelay;
        queue.Enqueue(
            (item.Client, item.Attempt + 1, nextDelay),
            nextTime);
    }

    completions.Sort();
    int peak = 0, peakIndex = 0;
    for (int i = 0; i < retryArrivals.Length; i++)
    {
        if (retryArrivals[i] > peak)
        {
            peak = retryArrivals[i];
            peakIndex = i;
        }
    }

    return new SimulationResult(
        name,
        completions.Count / (double)ClientCount,
        Percentile(completions, 0.50),
        Percentile(completions, 0.95),
        Percentile(completions, 0.99),
        peak,
        totalAttempts,
        totalAttempts / (double)ClientCount,
        peakIndex * BucketMs);
}

static int Percentile(List<int> sorted, double p)
{
    if (sorted.Count == 0) return 0;
    int index = (int)Math.Ceiling(p * sorted.Count) - 1;
    return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
}

readonly record struct RetryContext(int Client, int Attempt, double PreviousDelay)
{
    public int K => Attempt;
}

readonly record struct SimulationResult(
    string Name,
    double SuccessRate,
    int P50,
    int P95,
    int P99,
    int PeakRetryArrivals,
    long TotalAttempts,
    double MeanAttempts,
    int PeakAtMs);
