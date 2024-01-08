def decimal_to_percentage(decimal):
    return f'{decimal * 100:.2f}'.rstrip('0').rstrip('.') + '%'