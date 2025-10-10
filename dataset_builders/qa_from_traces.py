"""
Генератор Q&A данных из траекторий симуляции
Создает instruction-style пары для обучения LLM
"""

import json
import pandas as pd
from pathlib import Path
import random


class QAGenerator:
    """Генератор вопросов и ответов из траекторий"""
    
    def __init__(self, language='ru'):
        """
        Args:
            language: язык генерации ('ru' или 'en')
        """
        self.language = language
        
        # Шаблоны вопросов (русский)
        self.templates_ru = {
            'power_change': [
                "Мощность изменилась с {p1:.1f} до {p2:.1f} МВт за {dt:.1f} секунд. Почему это произошло?",
                "За {dt:.1f} секунд мощность {direction}. Объясните причину.",
                "Наблюдается {direction} мощности на {change:.1f} МВт. Какие факторы влияют на это?",
            ],
            'temperature': [
                "Температура топлива достигла {T:.1f}°C при мощности {P:.1f} МВт. Безопасно ли это?",
                "Почему температура теплоносителя {T:.1f}°C при расходе {flow:.2f}?",
                "Топливо нагрелось до {T:.1f}°C. Какие действия необходимы?",
            ],
            'control': [
                "Управляющие стержни находятся на позиции {pos:.1f}%. Какова текущая реактивность?",
                "Для достижения мощности {target:.1f} МВт нужно {action}. Почему?",
                "Стержни {motion}. Как это повлияет на мощность реактора?",
            ],
            'scram': [
                "При каких условиях был выполнен SCRAM на времени t={t:.1f}с?",
                "SCRAM сработал. Опишите последовательность событий.",
                "Температура топлива {T:.1f}°C. Нужен ли SCRAM?",
            ],
        }
    
    def generate_qa_pairs(self, trajectory_df, n_samples=100):
        """
        Сгенерировать Q&A пары из датафрейма траектории
        
        Args:
            trajectory_df: pandas DataFrame с траекторией
            n_samples: количество пар для генерации
        
        Returns:
            list of (question, answer) tuples
        """
        qa_pairs = []
        
        # Группируем по сценариям
        for scenario_id, group in trajectory_df.groupby('scenario_id'):
            if len(group) < 10:
                continue
            
            # Генерируем разные типы вопросов
            qa_pairs.extend(self._generate_power_qa(group))
            qa_pairs.extend(self._generate_temperature_qa(group))
            qa_pairs.extend(self._generate_control_qa(group))
            qa_pairs.extend(self._generate_safety_qa(group))
        
        # Семплируем нужное количество
        if len(qa_pairs) > n_samples:
            qa_pairs = random.sample(qa_pairs, n_samples)
        
        return qa_pairs
    
    def _generate_power_qa(self, df):
        """Генерация Q&A по изменению мощности"""
        qa = []
        
        if len(df) < 2:
            return qa
        
        # Находим значительные изменения мощности
        power = df['power_MW'].values
        for i in range(len(power) - 10):
            p1, p2 = power[i], power[i+10]
            change = p2 - p1
            
            if abs(change) > 5.0:  # Изменение больше 5 МВт
                dt = 10 * 0.001 * 10  # Примерное время
                direction = "выросла" if change > 0 else "упала"
                
                question = f"Мощность изменилась с {p1:.1f} до {p2:.1f} МВт за {dt:.1f} секунд. Почему это произошло?"
                
                # Анализ причины
                action = df.iloc[i:i+10]['action'].value_counts().index[0] if 'action' in df.columns else 'None'
                rod_pos_change = df.iloc[i+10]['rod_position_pct'] - df.iloc[i]['rod_position_pct']
                
                answer = f"Мощность {direction} на {abs(change):.1f} МВт. "
                if rod_pos_change > 1:
                    answer += "Управляющие стержни были извлечены, что увеличило реактивность и привело к росту мощности."
                elif rod_pos_change < -1:
                    answer += "Управляющие стержни были вставлены, что снизило реактивность и привело к падению мощности."
                else:
                    answer += "Изменение связано с температурной обратной связью или другими факторами."
                
                qa.append((question, answer))
                
                if len(qa) >= 5:  # Максимум 5 вопросов по одному сценарию
                    break
        
        return qa
    
    def _generate_temperature_qa(self, df):
        """Генерация Q&A по температуре"""
        qa = []
        
        # Находим высокие температуры
        for idx, row in df.iterrows():
            T_fuel = row['T_fuel_C']
            if T_fuel > 350 and random.random() < 0.1:  # 10% шанс
                question = f"Температура топлива достигла {T_fuel:.1f}°C при мощности {row['power_MW']:.1f} МВт. Безопасно ли это?"
                
                if T_fuel > 1200:
                    answer = "ОПАСНО! Температура превышает предел 1200°C. Необходим немедленный SCRAM."
                elif T_fuel > 900:
                    answer = "ПРЕДУПРЕЖДЕНИЕ! Температура приближается к опасной зоне. Рекомендуется снизить мощность."
                else:
                    answer = f"Температура в пределах нормы. Предел безопасности: 1200°C. Запас: {1200-T_fuel:.1f}°C."
                
                qa.append((question, answer))
                
                if len(qa) >= 3:
                    break
        
        return qa
    
    def _generate_control_qa(self, df):
        """Генерация Q&A по управлению"""
        qa = []
        
        # Анализ позиций стержней
        if len(df) > 0:
            sample = df.sample(min(3, len(df)))
            for idx, row in sample.iterrows():
                pos = row['rod_position_pct']
                reactivity = row['rod_reactivity_beta']
                
                question = f"Управляющие стержни находятся на позиции {pos:.1f}%. Какова текущая реактивность?"
                answer = f"При позиции стержней {pos:.1f}% реактивность составляет {reactivity:.3f} β. "
                
                if pos > 95:
                    answer += "Стержни почти полностью извлечены, реактор близок к критичности."
                elif pos < 20:
                    answer += "Стержни глубоко вставлены, реактор в глубоко докритическом состоянии."
                else:
                    answer += "Стержни в промежуточном положении, есть запас для маневра."
                
                qa.append((question, answer))
        
        return qa
    
    def _generate_safety_qa(self, df):
        """Генерация Q&A по безопасности"""
        qa = []
        
        # Поиск нарушений безопасности
        unsafe = df[~df['safe']]
        if len(unsafe) > 0:
            row = unsafe.iloc[0]
            
            question = f"На времени t={row['time']:.1f}с произошло нарушение безопасности. Что случилось?"
            
            answer = "Нарушение пределов безопасности: "
            if row['fuel_overheat']:
                answer += f"Перегрев топлива ({row['T_fuel_C']:.1f}°C > 1200°C). "
            if row['coolant_boiling']:
                answer += f"Кипение теплоносителя ({row['T_coolant_C']:.1f}°C > 320°C). "
            answer += "Был выполнен аварийный SCRAM для защиты реактора."
            
            qa.append((question, answer))
        
        return qa
    
    def save_qa_dataset(self, qa_pairs, output_path):
        """
        Сохранить Q&A датасет
        
        Args:
            qa_pairs: список пар (question, answer)
            output_path: путь для сохранения
        """
        records = []
        for i, (q, a) in enumerate(qa_pairs):
            records.append({
                'id': i,
                'question': q,
                'answer': a,
            })
        
        df = pd.DataFrame(records)
        
        # Сохраняем в JSONL (стандартный формат для LLM)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        print(f"✓ Q&A датасет сохранен: {output_path}")
        print(f"  Пар: {len(qa_pairs)}")
        
        return output_path


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Генератор Q&A из траекторий')
    parser.add_argument('--input', type=str, required=True, help='Путь к датасету траекторий (parquet/csv)')
    parser.add_argument('--output', type=str, default='./datasets/qa_dataset.jsonl', help='Путь для сохранения Q&A')
    parser.add_argument('--n-samples', type=int, default=100, help='Количество Q&A пар')
    
    args = parser.parse_args()
    
    print("Загрузка траекторий...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"  Загружено записей: {len(df)}")
    
    # Генерация Q&A
    print("Генерация Q&A пар...")
    generator = QAGenerator(language='ru')
    qa_pairs = generator.generate_qa_pairs(df, n_samples=args.n_samples)
    
    # Сохранение
    generator.save_qa_dataset(qa_pairs, args.output)
    
    # Вывод примеров
    print("\nПримеры сгенерированных Q&A:")
    for i, (q, a) in enumerate(qa_pairs[:3]):
        print(f"\n--- Пример {i+1} ---")
        print(f"Q: {q}")
        print(f"A: {a}")


if __name__ == '__main__':
    main()

