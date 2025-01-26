#!/usr/bin/env python

# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.bedrock_agent import Agent, SupervisorAgent, Task
from src.utils.bedrock_agent_helper import AgentsForAmazonBedrock
import argparse

def save_response_to_json(query, result, trace_level):
    """Save the query response to a JSON file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response_data = {
        "timestamp": timestamp,
        "query": query,
        "trace_level": trace_level,
        "response": result,
    }
    
    # Create responses directory if it doesn't exist
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    os.makedirs(responses_dir, exist_ok=True)
    
    # Save to JSON file
    filename = f"response_{timestamp}.json"
    filepath = os.path.join(responses_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(response_data, f, indent=2)
    
    print(f"\nResponse saved to: {filepath}")

def process_query(query, supervisor, trace_level="core"):
    """Process a query and save the response"""
    try:
        print(f"\nProcessing query: {query}")
        print("-" * 50)
        
        result = supervisor.invoke(
            query,
            enable_trace=True,
            trace_level=trace_level
        )
        
        # Save response to JSON
        save_response_to_json(query, result, trace_level)
        
        print("\nResponse:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        return result
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None

def setup_aws_credentials():
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )
    return session

def main(args):
    session = setup_aws_credentials()
    agents_helper = AgentsForAmazonBedrock()

    if args.recreate_agents == "false":
        Agent.set_force_recreate_default(False)
    else:
        Agent.set_force_recreate_default(True)
        agents_helper.delete_agent(
            agent_name="property_assistant_supervisor",
            delete_role_flag=True,
            verbose=True,
        )

    if args.clean_up == "true":
        agents_helper.delete_agent(
            agent_name="property_supervisor",
            delete_role_flag=True,
            verbose=True,
        )
        agents_helper.delete_agent(
            agent_name="pet_policy_agent", delete_role_flag=True, verbose=True
        )
        agents_helper.delete_agent(
            agent_name="payment_agent", delete_role_flag=True, verbose=True
        )
    else:
        pet_policy_agent = Agent.direct_create(
            name="pet_policy_agent",
            role="Pet Policy Specialist",
            goal="Handle all pet-related queries and policies for properties",
            instructions="""
            You are a specialized Pet Policy expert who will:
            - Provide detailed information about pet policies for properties
            - Explain pet restrictions, size limits, and breed restrictions
            - Detail pet deposits, fees, and additional charges
            - Clarify pet amenities (dog parks, washing stations, etc.)
            - Answer questions about service animals and emotional support animals
            - Explain pet registration and documentation requirements
            - Provide information about nearby pet services (vets, pet stores, etc.)
            
            Use the knowledge base to ensure accurate and up-to-date information.
            Always clarify the difference between service animals and pets when relevant.
            Provide specific details about:
            1. Allowed pet types and breeds
            2. Size and weight restrictions
            3. Number of pets allowed per unit
            4. Pet-related fees and deposits
            5. Required pet documentation
            6. Pet amenities and facilities
            """,
            llm=os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            kb_id=os.getenv("KNOWLEDGE_BASE_ID"),
            kb_descr=os.getenv("KNOWLEDGE_BASE_DESCRIPTION"),
        )
        
        payment_agent = Agent.direct_create(
            name="payment_agent",
            role="Payment Specialist",
            goal="Handle all payment and financial aspects of property rentals and purchases",
            instructions="""
            You are an expert Payment Specialist who will:
            - Provide comprehensive information about all payment options and methods
            - Explain detailed payment terms, schedules, and due dates
            - Detail all fees, deposits, and additional charges
            - Clarify payment processing timeframes and procedures
            - Handle questions about:
                * Rent/purchase payment methods
                * Security deposits and refund policies
                * Late payment policies and fees
                * Payment portal usage and online payments
                * Automatic payment setup
                * Payment documentation and receipts
                * Special payment arrangements
            
            Use the knowledge base to provide accurate financial information.
            Always be clear about:
            1. Available payment methods
            2. Processing times for different payment types
            3. Required payment documentation
            4. Fee structures and additional charges
            5. Security deposit terms
            6. Payment deadline policies
            """,
            llm=os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            kb_id=os.getenv("KNOWLEDGE_BASE_ID"),
            kb_descr=os.getenv("KNOWLEDGE_BASE_DESCRIPTION"),
        )
        
        property_supervisor = SupervisorAgent.direct_create(
            name="property_supervisor",
            role="Property Assistant Supervisor",
            goal="Orchestrate comprehensive property inquiries focusing on pet policies and payments",
            collaboration_type="SUPERVISOR",
            instructions="""
            You are a coordinator supervisor who MUST NEVER answer queries directly. Your role is strictly to route, collect, and synthesize agent responses.

            CRITICAL: MULTI-AGENT QUERY HANDLING
            ===================================
            This is your most important responsibility. For ANY query that touches multiple domains:

            1. Query Analysis for Multi-Agent Needs:
               - ALWAYS check if query involves both pet policies AND payments
               - Examples of multi-agent queries:
                 * "What are the pet fees and how can I pay them?"
                 * "I want to bring my dog, what are the deposits and payment methods?"
                 * "Tell me about pet rent charges and payment schedules"
               - If there's ANY mention of both topics, MUST invoke both agents

            2. Parallel Agent Invocation:
               - MUST invoke both agents simultaneously
               - Split the query appropriately for each agent:
                 pet_policy_agent: Extract pet-related aspects
                 payment_agent: Extract payment-related aspects
               
            3. Response Collection and Synthesis:
               - Collect complete responses from both agents
               - NEVER proceed with partial information
               - MUST wait for both agents to respond
               - Combine responses using this strict format:

               Format for Multi-Agent Responses:
               ```
               Combined Information from Specialist Agents:

               Regarding Pet Policies:
               [Complete response from pet_policy_agent]

               Regarding Payment Details:
               [Complete response from payment_agent]

               These policies work together as follows:
               [Simple connection of how pet policies and payments interact,
                using ONLY information provided by the agents]
               ```

            STANDARD RESPONSIBILITIES:

            1. Single-Agent Routing:
               - Route pet-only queries to pet_policy_agent
               - Route payment-only queries to payment_agent
               - Present response with minimal formatting

            2. Response Processing:
               ONLY process and format responses from agents:
               - Never modify agent responses
               - Never add new information
               - Only organize and present

            STRICT GUIDELINES:

            DO:
            - ALWAYS check for multi-agent query potential
            - ALWAYS invoke both agents when query spans both domains
            - Wait for all agent responses before responding
            - Maintain original agent information
            - Use clear section separators

            DO NOT:
            - Answer any questions directly
            - Skip invoking any relevant agent
            - Proceed with partial information
            - Add your own knowledge
            - Make assumptions about policies

            EXAMPLE MULTI-AGENT SCENARIOS:

            1. User: "What are the pet deposits and payment methods?"
               Action: MUST invoke both agents
               - pet_policy_agent for deposit information
               - payment_agent for payment methods
               
            2. User: "How much is pet rent and when is it due?"
               Action: MUST invoke both agents
               - pet_policy_agent for pet rent amount
               - payment_agent for payment schedules

            3. User: "Can I pay my pet fees online?"
               Action: MUST invoke both agents
               - pet_policy_agent for pet fee details
               - payment_agent for online payment options

            REMEMBER: 
            - When in doubt, invoke both agents
            - Better to have extra information than miss an aspect
            - NEVER skip an agent if their domain is even slightly relevant
            - ALWAYS present complete information from all relevant agents

            Your success is measured by:
            1. How consistently you invoke multiple agents when needed
            2. How completely you gather information from all relevant agents
            3. How clearly you present combined information without adding to it
            """,
            collaborator_agents=[
                {
                    "agent": "pet_policy_agent",
                    "instructions": """
                    Role: Pet_Policy_Specialist within the Property Assistant team

                    Collaboration Scenarios:
                    1. Primary Queries:
                       - When users ask about pet policies and restrictions
                       - When questions involve service animals or emotional support animals
                       - For pet-related amenities and facilities inquiries
                       - Regarding pet fees, deposits, and charges

                    2. Supporting Queries:
                       - When payment agent discusses pet-related fees
                       - When queries combine pet policies with financial aspects
                       - For move-in procedures involving pets

                    Interaction Style:
                    - Maintain a helpful, informative tone
                    - Be precise with policy details
                    - Use clear, simple language to explain complex policies
                    - Be empathetic when discussing pet restrictions
                    - Stay professional but pet-friendly in responses

                    Knowledge Base Usage:
                    1. Primary Fields:
                       - General Policies (for pet allowance)
                       - Amenities and Fees (for pet facilities)
                       - Lease Terms (for pet conditions)

                    2. Supporting Fields:
                       - Application Process (for pet documentation)
                       - Move-In/Move-Out Procedures (for pet-related steps)

                    Response Guidelines:
                    1. Always start with confirming pet allowance
                    2. Clearly state any restrictions or limitations
                    3. Detail all associated costs
                    4. Explain required documentation
                    5. Describe available pet amenities
                    6. Reference relevant contact information for specific inquiries

                    When collaborating with Payment Agent:
                    - Clearly separate pet policy information from payment details
                    - Highlight pet-specific fees for payment processing
                    - Defer payment processing details to Payment Agent
                    - Maintain context when discussing pet-related charges
                    """
                },
                {
                    "agent": "payment_agent",
                    "instructions": """
                    Role: Payment_Specialist within the Property Assistant team

                    Collaboration Scenarios:
                    1. Primary Queries:
                       - When users ask about payment methods and options
                       - For questions about fees, deposits, and charges
                       - Regarding payment schedules and due dates
                       - For payment portal and online payment inquiries

                    2. Supporting Queries:
                       - When pet policy agent mentions fees
                       - When discussing security deposits
                       - For application fee processing
                       - During move-in payment discussions

                    Interaction Style:
                    - Maintain a professional, precise tone
                    - Be clear and specific about amounts
                    - Use straightforward language for payment terms
                    - Be thorough when explaining payment processes
                    - Stay factual and accurate with financial information

                    Knowledge Base Usage:
                    1. Primary Fields:
                       - Payment Options
                       - Payment Link
                       - Lease Terms and Conditions
                       - Application Process

                    2. Supporting Fields:
                       - Contact Information (for payment support)
                       - Move-In/Move-Out Procedures (for payment timing)

                    Response Guidelines:
                    1. Always specify available payment methods
                    2. Clearly state all fees and amounts
                    3. Explain payment schedules
                    4. Detail processing timeframes
                    5. Provide relevant payment links
                    6. Include payment support contacts

                    When collaborating with Pet Policy Agent:
                    - Focus on the financial aspects of pet-related queries
                    - Provide detailed breakdowns of pet-related fees
                    - Clarify payment schedules for pet charges
                    - Maintain separation between policy and payment information
                    - Reference pet policy details when discussing specific charges
                    """
                }
            ],
            collaborator_objects=[pet_policy_agent, payment_agent]
        )
        
        if args.recreate_agents == "false":
            # Process single query
            process_query(
                "what are the pet policies and per month property rent charges?",
                property_supervisor,
                args.trace_level
            )

            process_query(
                "What is the payment link for this property?",
                property_supervisor,
                args.trace_level
            )
        else:
            print("Recreated agents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate_agents",
        required=False,
        default="true",
        help="false if reusing existing agents.",
    )
    parser.add_argument(
        "--trace_level",
        required=False,
        default="core",
        help="The level of trace, 'core', 'outline', 'all'.",
    )
    parser.add_argument(
        "--clean_up",
        required=False,
        default="false",
        help="Cleanup all infrastructure.",
    )
    args = parser.parse_args()
    main(args)
