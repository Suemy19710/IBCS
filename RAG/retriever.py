from RAG.ibcs_docs import ibcs_docs

def get_ibcs_rule(rule_name):
    for rule in ibcs_docs:
        if rule["rule"].lower() == rule_name.lower():
            return rule
    return None