import type { ReactNode } from 'react';

type MetricVariant = 'primary' | 'success' | 'warning' | 'danger' | 'neutral';

interface MetricCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon?: ReactNode;
  variant?: MetricVariant;
}

const VARIANT_STYLES: Record<MetricVariant, string> = {
  primary:
    'gradient-primary text-white shadow-card hover:shadow-card/80 card-hover',
  success:
    'gradient-success text-white shadow-card hover:shadow-card/80 card-hover',
  warning: 'bg-white text-slate-900 border border-warning/30 card-hover',
  danger: 'bg-white text-slate-900 border border-danger/30 card-hover',
  neutral: 'bg-white text-slate-900 card-hover',
};

const MetricCard = ({
  title,
  value,
  subtitle,
  icon,
  variant = 'neutral',
}: MetricCardProps) => {
  const isGradient = variant === 'primary' || variant === 'success';
  const titleColor = isGradient ? 'text-white/80' : 'text-slate-500';
  const valueColor = isGradient ? 'text-white' : 'text-slate-900';
  const subtitleColor = isGradient ? 'text-white/75' : 'text-slate-500';

  return (
    <article
      className={`relative overflow-hidden rounded-2xl p-6 transition duration-250 ${VARIANT_STYLES[variant]}`}
      role="presentation"
    >
      {icon ? (
        <div className="absolute right-6 top-6 text-white/40">{icon}</div>
      ) : null}
      <p
        className={`text-xs font-semibold uppercase tracking-[0.2em] ${titleColor}`}
      >
        {title}
      </p>
      <p className={`mt-3 text-3xl font-bold ${valueColor}`}>{value}</p>
      {subtitle ? (
        <p className={`mt-3 text-sm ${subtitleColor}`}>{subtitle}</p>
      ) : null}
    </article>
  );
};

export default MetricCard;
