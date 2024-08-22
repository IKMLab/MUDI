def parse_convai2_data(data_file):
    with open(data_file, encoding='utf8') as f:
        persona = []
        query = []
        response = []
        cand = []
        is_persona = False
        tmp_persona = []
        tmp_query = []
        tmp_response = []
        tmp_cand = []
        first = True
        cnt = 0
        sum_u = 0
        for line in f:
            cnt += 1
            line = line.strip()
            if 'your persona: ' in line:
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []
                first = False
                is_persona = True
                line = line.split(': ', maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []
                line = line[line.find(' ') + 1:]
                tmp_query.append(line.split('\t')[0])
                tmp_response.append(line.split('\t')[1])
                tmp_cand.append(line.split('\t')[3].split('|'))
        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)
        assert len(query) == len(response) == len(persona) == len(cand)

    print(f'{data_file} has {len(query)} dialog and {sum_u} query')

    return persona, query, response, cand
